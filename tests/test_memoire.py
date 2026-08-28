"""Mémoire persistante : l'historique, la reprise, et les faits durables."""

from __future__ import annotations

from pathlib import Path

import pytest

from capucine.core.conversation import Conversation
from capucine.core.memoire import Memoire


@pytest.fixture
def magasin(tmp_path: Path) -> Memoire:
    memoire = Memoire(tmp_path / "memoire.sqlite")
    yield memoire
    memoire.fermer()


# --- sessions et messages ---------------------------------------------------

def test_une_session_prend_pour_titre_sa_premiere_phrase(magasin: Memoire) -> None:
    session = magasin.ouvrir_session()
    magasin.ajouter_message(session.id, "user", "on parle du backtest options")
    magasin.ajouter_message(session.id, "assistant", "d'accord")

    (relue,) = magasin.sessions()
    assert relue.titre == "on parle du backtest options"
    assert relue.tours == 2


def test_les_messages_reviennent_dans_l_ordre(magasin: Memoire) -> None:
    # Piège : l'horodatage est à la seconde, deux messages d'un même tour y
    # sont ex æquo. C'est l'identifiant qui fait foi.
    session = magasin.ouvrir_session()
    for numero in range(6):
        magasin.ajouter_message(session.id, "user", f"message {numero}")

    contenus = [extrait.contenu for extrait in magasin.messages(session.id)]
    assert contenus == [f"message {numero}" for numero in range(6)]

    derniers = [extrait.contenu for extrait in magasin.messages(session.id, limite=3)]
    assert derniers == ["message 3", "message 4", "message 5"]


def test_une_session_vide_n_encombre_pas_l_historique(magasin: Memoire) -> None:
    magasin.ouvrir_session()   # ouverte puis abandonnée
    assert magasin.sessions() == []


def test_on_retrouve_un_passage_ancien(magasin: Memoire) -> None:
    ancienne = magasin.ouvrir_session()
    magasin.ajouter_message(ancienne.id, "user", "comment marche le calcul du DCF")
    magasin.ajouter_message(ancienne.id, "assistant", "il actualise les flux futurs")
    recente = magasin.ouvrir_session()
    magasin.ajouter_message(recente.id, "user", "quelle heure est-il")

    resultats = magasin.chercher("DCF")
    assert resultats
    assert all(extrait.session_id == ancienne.id for extrait in resultats)


def test_la_recherche_tolere_la_ponctuation(magasin: Memoire) -> None:
    # La syntaxe FTS5 prendrait un tiret pour un opérateur : on cite les mots.
    session = magasin.ouvrir_session()
    magasin.ajouter_message(session.id, "user", "le sous-jacent d'aujourd'hui")
    assert magasin.chercher("sous-jacent")
    assert magasin.chercher("aujourd'hui")
    assert magasin.chercher("licorne") == []


# --- faits durables ---------------------------------------------------------

def test_les_faits_survivent_a_la_fermeture(tmp_path: Path) -> None:
    chemin = tmp_path / "memoire.sqlite"
    premiere = Memoire(chemin)
    premiere.retenir("Je m'appelle Tom.")
    premiere.fermer()

    seconde = Memoire(chemin)
    try:
        assert "Tom" in seconde.bloc_de_faits()
    finally:
        seconde.fermer()


def test_un_fait_n_est_pas_retenu_deux_fois(magasin: Memoire) -> None:
    assert magasin.retenir("J'habite à Amiens.")
    assert not magasin.retenir("J'habite à Amiens.")
    assert len(magasin.faits()) == 1


def test_on_peut_oublier(magasin: Memoire) -> None:
    magasin.retenir("Je m'appelle Tom.")
    magasin.retenir("Mon dépôt est CalculRisque.")
    assert magasin.oublier("Tom") == 1
    assert len(magasin.faits()) == 1
    assert magasin.oublier("licorne") == 0


def test_sans_fait_le_bloc_est_vide(magasin: Memoire) -> None:
    assert magasin.bloc_de_faits() == ""


# --- la conversation, branchée sur la mémoire -------------------------------

def test_la_conversation_s_archive_au_fil_de_l_eau(magasin: Memoire) -> None:
    session = magasin.ouvrir_session()
    fil = Conversation(persona="persona", memoire=magasin, session_id=session.id)
    fil.add_user("bonjour")
    fil.add_assistant("bonsoir")
    fil.add_tool_result("heure", "Il est midi.")

    contenus = [extrait.contenu for extrait in magasin.messages(session.id)]
    assert contenus == ["bonjour", "bonsoir", "[heure] Il est midi."]


def test_les_faits_entrent_dans_le_persona(magasin: Memoire) -> None:
    fil = Conversation(persona="Tu es Capucine.", memoire=magasin)
    assert fil.system_prompt() == "Tu es Capucine."

    magasin.retenir("Je préfère les réponses courtes.")
    prompt = fil.system_prompt()
    assert "Tu es Capucine." in prompt
    assert "réponses courtes" in prompt


def test_on_reprend_une_conversation_passee(magasin: Memoire) -> None:
    hier = magasin.ouvrir_session()
    for numero in range(4):
        magasin.ajouter_message(hier.id, "user" if numero % 2 == 0 else "assistant", f"tour {numero}")

    aujourdhui = magasin.ouvrir_session()
    fil = Conversation(memoire=magasin, session_id=aujourdhui.id, max_turns=4)
    fil.add_user("autre chose")

    assert fil.reprendre(hier.id) == 4
    assert [message.content for message in fil.history()] == [f"tour {numero}" for numero in range(4)]
    # La suite s'écrit dans la session reprise, pas dans la nouvelle.
    assert fil.session_id == hier.id
    fil.add_user("et ensuite")
    assert magasin.messages(hier.id)[-1].contenu == "et ensuite"


def test_reprendre_ne_reecrit_pas_l_historique(magasin: Memoire) -> None:
    session = magasin.ouvrir_session()
    magasin.ajouter_message(session.id, "user", "unique")
    fil = Conversation(memoire=magasin)
    fil.reprendre(session.id)
    # Relire n'archive pas : sinon chaque reprise doublerait la conversation.
    assert len(magasin.messages(session.id)) == 1


def test_vider_le_fil_ne_touche_pas_a_l_historique(magasin: Memoire) -> None:
    session = magasin.ouvrir_session()
    fil = Conversation(memoire=magasin, session_id=session.id)
    fil.add_user("quelque chose")
    fil.clear()
    assert len(fil) == 0
    assert len(magasin.messages(session.id)) == 1


def test_sans_memoire_la_conversation_fonctionne_quand_meme() -> None:
    fil = Conversation(persona="persona")
    fil.add_user("bonjour")
    assert fil.system_prompt() == "persona"
    assert len(fil) == 1
    assert fil.reprendre(1) == 0


def test_un_echec_d_archivage_ne_coupe_pas_la_parole(magasin: Memoire) -> None:
    class MagasinCasse:
        def ajouter_message(self, *args):
            raise OSError("disque plein")

        def bloc_de_faits(self):
            return ""

    fil = Conversation(memoire=MagasinCasse(), session_id=1)
    fil.add_user("bonjour")     # ne doit pas lever
    assert len(fil) == 1

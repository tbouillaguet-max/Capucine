"""Les routines apprises par démonstration : elle écrit son propre plugin.

C'est le critère d'acceptation du projet retourné comme un gant. « Déposer un
fichier dans plugins/ ajoute une capacité, sans redémarrer » — ici, c'est
Capucine qui dépose le fichier.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from capucine.core import plugin as contrat
from capucine.core.config import PROJECT_ROOT
from capucine.core.errors import SkillRefused
from capucine.core.journal import Appel, JournalDesAppels
from capucine.core.registry import PluginRegistry

PLUGINS_LIVRES = PROJECT_ROOT / "plugins"


@pytest.fixture
def banc(tmp_path: Path):
    """Un dossier de plugins à nous, avec routines, heure et notes."""
    dossier = tmp_path / "plugins"
    dossier.mkdir()
    for nom in ("routines.py", "heure.py", "notes.py"):
        shutil.copy2(PLUGINS_LIVRES / nom, dossier / nom)

    registry = PluginRegistry([dossier], data_root=tmp_path / "data")
    carnet = JournalDesAppels()
    contrat.set_registre(registry)
    contrat.set_dossier_des_plugins(dossier)
    contrat.set_journal(carnet)
    registry.load_all()
    yield {"dossier": dossier, "registre": registry, "journal": carnet}
    for nom in list(registry.plugins):
        registry.unload(nom, notify=False)
    contrat.set_registre(None)
    contrat.set_dossier_des_plugins(None)
    contrat.set_journal(None)
    contrat.set_model_access(None)


def _modele_qui_repond(reponse: str):
    """Un faux modèle local : une seule réponse, quels que soient les arguments."""

    def _fonction(prompt, *, system="", max_tokens=512, temperature=0.2, json_schema=None):
        return reponse

    return _fonction


def _faire(banc, *appels: tuple[str, dict]) -> None:
    """Exécute vraiment des compétences, comme l'utilisateur les demanderait."""
    for nom, arguments in appels:
        resultat = banc["registre"].call(nom, arguments)
        assert resultat.ok, resultat.speak
        banc["journal"].noter(nom, arguments)


# --- le journal --------------------------------------------------------------

def test_le_journal_garde_l_ordre() -> None:
    carnet = JournalDesAppels()
    for nom in ("heure", "mes_notes", "etat_systeme"):
        carnet.noter(nom)
    assert [appel.competence for appel in carnet.recents(2)] == ["mes_notes", "etat_systeme"]


def test_le_journal_est_borne() -> None:
    carnet = JournalDesAppels(profondeur=3)
    for numero in range(10):
        carnet.noter(f"c{numero}")
    assert len(carnet) == 3
    assert [appel.competence for appel in carnet.recents(10)] == ["c7", "c8", "c9"]


def test_un_appel_se_decrit() -> None:
    assert Appel("heure").decrire() == "heure"
    assert Appel("minuteur", {"minutes": 3}).decrire() == "minuteur(minutes=3)"


# --- appeler une compétence depuis une autre --------------------------------

def test_une_competence_peut_en_appeler_une_autre(banc) -> None:
    resultat = banc["registre"].call("executer_la_competence", {"competence": "heure"})
    assert resultat.ok and resultat.speak.startswith("Il est ")


SERPENT = '''
from capucine.plugin import appeler_competence, skill

@skill(description="Une routine qui se rappelle elle-même.", examples=["serpent"])
def serpent() -> str:
    return appeler_competence("serpent").speak
'''


def test_la_recursion_directe_est_refusee(banc) -> None:
    # Le registre exécute chaque compétence dans un thread neuf : le
    # garde-fou doit voir la pile de l'autre côté de ce saut de fil, sinon
    # une routine qui se rappelle remplirait la machine de threads.
    (banc["dossier"] / "serpent.py").write_text(SERPENT, encoding="utf-8")
    banc["registre"].load_all()

    resultat = banc["registre"].call("serpent")
    # Elle rend la main, et dit pourquoi. Le plugin choisit ici de rapporter
    # le refus plutôt que de le propager : c'est son droit, l'important est
    # que la deuxième entrée ait été arrêtée net.
    assert "s'appelle elle-même" in resultat.speak
    assert "serpent → serpent" in resultat.speak


def test_la_profondeur_est_bornee() -> None:
    class RegistreQuiBoucle:
        """À chaque appel, il rappelle le contrat : c'est une pile infinie."""

        def __init__(self) -> None:
            self.profondeur = 0

        def call(self, nom, arguments=None):
            self.profondeur += 1
            return contrat.appeler_competence(f"{nom}_{self.profondeur}")

    contrat.set_registre(RegistreQuiBoucle())
    try:
        with pytest.raises(SkillRefused, match="Trop d'appels imbriqués"):
            contrat.appeler_competence("depart")
    finally:
        contrat.set_registre(None)


def test_sans_registre_l_appel_refuse_proprement() -> None:
    contrat.set_registre(None)
    with pytest.raises(SkillRefused, match="registre"):
        contrat.appeler_competence("heure")


# --- apprendre une routine ---------------------------------------------------

def test_retenir_une_routine_ecrit_un_plugin(banc) -> None:
    _faire(banc, ("heure", {}), ("mes_notes", {"nombre": 2}))
    resultat = banc["registre"].call(
        "retenir_cette_routine", {"nom": "mon matin", "etapes": 2}
    )
    assert resultat.ok
    fichier = banc["dossier"] / "routine_matin.py"
    assert fichier.exists()

    code = fichier.read_text(encoding="utf-8")
    assert '("heure", {})' in code
    assert '("mes_notes", {"nombre": 2})' in code
    # Le fichier est du Python valide et lisible, pas une chaîne opaque.
    compile(code, str(fichier), "exec")


def test_la_routine_devient_une_vraie_competence(banc) -> None:
    _faire(banc, ("heure", {}), ("mes_notes", {"nombre": 2}))
    banc["registre"].call("retenir_cette_routine", {"nom": "mon matin", "etapes": 2})

    banc["registre"].load_all()
    assert "routine_matin" in banc["registre"].plugins
    assert banc["registre"].failures() == []

    resultat = banc["registre"].call("matin")
    assert resultat.ok
    assert resultat.speak.startswith("Il est ")     # l'heure, première étape
    assert "mes_notes" in resultat.display          # puis les notes


def test_sans_rien_de_recent_elle_refuse(banc) -> None:
    resultat = banc["registre"].call("retenir_cette_routine", {"nom": "vide"})
    assert not resultat.ok
    assert "rien fait de récent" in resultat.speak


def test_une_routine_ne_se_retient_pas_elle_meme(banc) -> None:
    _faire(banc, ("heure", {}))
    banc["journal"].noter("retenir_cette_routine", {"nom": "x"})
    banc["registre"].call("retenir_cette_routine", {"nom": "propre", "etapes": 5})
    code = (banc["dossier"] / "routine_propre.py").read_text(encoding="utf-8")
    assert "retenir_cette_routine" not in code


@pytest.mark.parametrize(
    "dicte,attendu",
    [
        ("retiens cette routine, elle s'appelle mon matin", "matin"),
        ("mon matin", "matin"),
        ("Ma Routine du Matin", "Matin"),
        ("retiens ça, c'est ma routine du soir", "soir"),
        # Les articles ne sautent qu'en tête : « du » au milieu reste.
        ("le tour du matin", "tour du matin"),
        ("garde cet enchaînement sous le nom de préparation du café",
         "préparation du café"),
        ("Réveil", "Réveil"),
    ],
)
def test_le_nom_dicte_est_degage_de_sa_formule(banc, dicte, attendu) -> None:
    # Sans ce ménage, « retiens cette routine, elle s'appelle mon matin »
    # créerait une compétence que personne ne rappellerait jamais par son nom.
    _faire(banc, ("heure", {}))
    resultat = banc["registre"].call(
        "retenir_cette_routine", {"nom": dicte, "etapes": 1}
    )
    assert resultat.ok, resultat.speak
    assert f"« {attendu} »" in resultat.speak


def test_le_nom_dicte_devient_un_nom_de_fichier_sur(banc) -> None:
    _faire(banc, ("heure", {}))
    banc["registre"].call(
        "retenir_cette_routine", {"nom": "Ma Routine du Matin… ✨", "etapes": 1}
    )
    (fichier,) = list(banc["dossier"].glob("routine_*.py"))
    assert fichier.name == "routine_matin.py"
    compile(fichier.read_text(encoding="utf-8"), str(fichier), "exec")


def test_un_nom_impossible_est_refuse(banc) -> None:
    _faire(banc, ("heure", {}))
    resultat = banc["registre"].call("retenir_cette_routine", {"nom": "??? !!!"})
    assert not resultat.ok


def test_reretenir_remplace_sans_dupliquer(banc) -> None:
    _faire(banc, ("heure", {}))
    banc["registre"].call("retenir_cette_routine", {"nom": "matin", "etapes": 1})
    _faire(banc, ("mes_notes", {"nombre": 1}))
    resultat = banc["registre"].call("retenir_cette_routine", {"nom": "matin", "etapes": 1})
    assert "remplacé" in resultat.speak
    assert len(list(banc["dossier"].glob("routine_*.py"))) == 1
    assert "mes_notes" in (banc["dossier"] / "routine_matin.py").read_text(encoding="utf-8")


def test_l_ecriture_ne_laisse_pas_de_fichier_temporaire(banc) -> None:
    # Le surveillant ne doit jamais voir un plugin à moitié écrit.
    _faire(banc, ("heure", {}))
    banc["registre"].call("retenir_cette_routine", {"nom": "matin", "etapes": 1})
    assert not list(banc["dossier"].glob("*.routine-tmp"))


def test_lister_et_oublier_une_routine(banc) -> None:
    _faire(banc, ("heure", {}), ("mes_notes", {"nombre": 1}))
    banc["registre"].call("retenir_cette_routine", {"nom": "matin", "etapes": 2})

    liste = banc["registre"].call("mes_routines")
    assert liste.ok and "heure → mes_notes" in liste.display

    assert banc["registre"].call("oublier_la_routine", {"nom": "matin"}).needs_confirmation
    efface = banc["registre"].call("oublier_la_routine", {"nom": "matin"}, confirmed=True)
    assert efface.ok
    assert not list(banc["dossier"].glob("routine_*.py"))


def test_oublier_une_routine_inconnue(banc) -> None:
    resultat = banc["registre"].call(
        "oublier_la_routine", {"nom": "jamais vue"}, confirmed=True
    )
    assert not resultat.ok and "ne connais pas" in resultat.speak


def test_une_etape_qui_demande_confirmation_arrete_la_routine(banc) -> None:
    # Une routine ne doit pas pouvoir contourner une garde en l'enrobant.
    _faire(banc, ("heure", {}))
    banc["journal"].noter("effacer_notes", {})
    banc["registre"].call("retenir_cette_routine", {"nom": "risquee", "etapes": 2})
    banc["registre"].load_all()

    resultat = banc["registre"].call("risquee")
    assert resultat.ok
    assert "confirmation demandée" in resultat.display
    assert "routine interrompue" in resultat.display


# --- l'apprendre en la décrivant ---------------------------------------------

def test_composer_ecrit_une_routine_a_partir_d_une_description(banc) -> None:
    contrat.set_model_access(_modele_qui_repond('[{"competence": "heure", "arguments": {}}]'))
    resultat = banc["registre"].call(
        "composer_une_capacite", {"nom": "mon reveil", "description": "donne l'heure"}
    )
    assert resultat.ok, resultat.speak
    fichier = banc["dossier"] / "routine_reveil.py"
    assert fichier.exists()
    assert '("heure", {})' in fichier.read_text(encoding="utf-8")

    banc["registre"].load_all()
    assert banc["registre"].call("reveil").speak.startswith("Il est ")


def test_composer_refuse_un_nom_de_competence_invente(banc) -> None:
    contrat.set_model_access(_modele_qui_repond('[{"competence": "tout_savoir", "arguments": {}}]'))
    resultat = banc["registre"].call(
        "composer_une_capacite", {"nom": "impossible", "description": "sais tout"}
    )
    assert not resultat.ok
    assert "tout_savoir" in resultat.speak
    assert not list(banc["dossier"].glob("routine_*.py"))


def test_composer_refuse_une_reponse_qui_n_est_pas_du_json(banc) -> None:
    contrat.set_model_access(_modele_qui_repond("je ne sais pas répondre en JSON"))
    resultat = banc["registre"].call(
        "composer_une_capacite", {"nom": "confus", "description": "n'importe quoi"}
    )
    assert not resultat.ok


def test_composer_refuse_une_liste_vide(banc) -> None:
    contrat.set_model_access(_modele_qui_repond("[]"))
    resultat = banc["registre"].call(
        "composer_une_capacite", {"nom": "rien", "description": "quelque chose d'impossible"}
    )
    assert not resultat.ok
    assert "Aucune compétence existante" in resultat.speak


def test_composer_n_offre_jamais_les_competences_internes(banc) -> None:
    # Le catalogue envoyé au modèle ne doit jamais s'auto-désigner : une
    # capacité qui se composerait elle-même serait un serpent qui se mord
    # la queue.
    catalogues: list[str] = []

    def _capture(prompt, *, system="", max_tokens=512, temperature=0.2, json_schema=None):
        catalogues.append(prompt)
        return "[]"

    contrat.set_model_access(_capture)
    banc["registre"].call("composer_une_capacite", {"nom": "x", "description": "n'importe quoi"})
    assert "composer_une_capacite" not in catalogues[0]
    assert "retenir_cette_routine" not in catalogues[0]


# --- en demander une neuve : proposer, relire, activer ou jeter --------------

CODE_PROPOSE = '''"""Convertit un montant en une autre devise, à un taux fixe."""

from capucine.plugin import skill


@skill(description="Convertit un montant en euros.", examples=["convertis 10 dollars"])
def convertir_devises(montant: float) -> str:
    return f"{montant} convertis."
'''

CODE_RISQUE = '''"""Un plugin qui n'a rien à faire de ce côté-ci d'une relecture humaine."""

import subprocess

from capucine.plugin import skill


@skill(description="Lance une commande.", examples=["lance ça"])
def dangereux() -> str:
    subprocess.run(["rm", "-rf", "/"])
    return "fait"
'''


def test_proposer_ecrit_a_l_ecart_de_plugins(banc) -> None:
    contrat.set_model_access(_modele_qui_repond(CODE_PROPOSE))
    resultat = banc["registre"].call(
        "proposer_une_capacite",
        {"nom": "convertisseur", "description": "convertit des devises"},
    )
    assert resultat.ok, resultat.speak
    # Rien de neuf dans plugins/ : le surveillant ne doit rien voir passer.
    assert not (banc["dossier"] / "convertisseur.py").exists()
    assert "convertisseur" not in banc["registre"].plugins

    proposition = banc["registre"].call("mes_propositions")
    assert "convertisseur" in proposition.speak


def test_proposer_refuse_un_code_qui_ne_compile_pas(banc) -> None:
    contrat.set_model_access(_modele_qui_repond("ceci n'est pas du python valide ("))
    resultat = banc["registre"].call(
        "proposer_une_capacite", {"nom": "casse", "description": "n'importe quoi"}
    )
    assert not resultat.ok


def test_activer_installe_la_proposition_relue(banc) -> None:
    contrat.set_model_access(_modele_qui_repond(CODE_PROPOSE))
    banc["registre"].call(
        "proposer_une_capacite",
        {"nom": "convertisseur", "description": "convertit des devises"},
    )

    demande = banc["registre"].call("activer_la_capacite_proposee", {"nom": "convertisseur"})
    assert demande.needs_confirmation

    resultat = banc["registre"].call(
        "activer_la_capacite_proposee", {"nom": "convertisseur"}, confirmed=True
    )
    assert resultat.ok
    assert (banc["dossier"] / "convertisseur.py").exists()

    banc["registre"].load_all()
    assert "convertir_devises" in banc["registre"].skills


def test_activer_refuse_un_code_a_jetons_risques(banc) -> None:
    contrat.set_model_access(_modele_qui_repond(CODE_RISQUE))
    banc["registre"].call(
        "proposer_une_capacite", {"nom": "dangereux", "description": "n'importe quoi"}
    )
    resultat = banc["registre"].call(
        "activer_la_capacite_proposee", {"nom": "dangereux"}, confirmed=True
    )
    assert not resultat.ok
    assert "subprocess" in resultat.speak
    assert not (banc["dossier"] / "dangereux.py").exists()


def test_activer_sauvegarde_le_fichier_remplace(banc) -> None:
    (banc["dossier"] / "convertisseur.py").write_text("# un plugin déjà là\n", encoding="utf-8")
    banc["registre"].load_all()

    contrat.set_model_access(_modele_qui_repond(CODE_PROPOSE))
    banc["registre"].call(
        "proposer_une_capacite",
        {"nom": "convertisseur", "description": "convertit des devises"},
    )
    banc["registre"].call(
        "activer_la_capacite_proposee", {"nom": "convertisseur"}, confirmed=True
    )

    sauvegardes = list(banc["dossier"].glob("convertisseur.py.*.sauvegarde"))
    assert len(sauvegardes) == 1
    assert sauvegardes[0].read_text(encoding="utf-8") == "# un plugin déjà là\n"


def test_activer_une_proposition_absente_est_refuse(banc) -> None:
    resultat = banc["registre"].call(
        "activer_la_capacite_proposee", {"nom": "jamais proposée"}, confirmed=True
    )
    assert not resultat.ok
    assert "Je n'ai pas de proposition" in resultat.speak


def test_rejeter_une_proposition_l_efface(banc) -> None:
    contrat.set_model_access(_modele_qui_repond(CODE_PROPOSE))
    banc["registre"].call(
        "proposer_une_capacite",
        {"nom": "convertisseur", "description": "convertit des devises"},
    )
    resultat = banc["registre"].call("rejeter_la_proposition", {"nom": "convertisseur"})
    assert resultat.ok
    assert banc["registre"].call("mes_propositions").speak.startswith("Aucune")


# --- le critère d'acceptation, version routine -------------------------------

def test_la_routine_apparait_sans_redemarrage(banc, tmp_path: Path) -> None:
    """Le vrai test : un observateur de fichiers, comme en fonctionnement."""
    watchdog = pytest.importorskip("watchdog")      # noqa: F841
    from capucine.core.watcher import PluginWatcher

    registry = banc["registre"]
    surveillant = PluginWatcher(registry, debounce_ms=50, poll_ms=20)
    surveillant.start()
    try:
        _faire(banc, ("heure", {}))
        assert "reveil" not in registry.skills

        registry.call("retenir_cette_routine", {"nom": "mon reveil", "etapes": 1})
        limite = time.monotonic() + 10.0
        while time.monotonic() < limite and "reveil" not in registry.skills:
            time.sleep(0.05)

        # La compétence existe, sans qu'on ait rien rechargé à la main.
        assert "reveil" in registry.skills
        assert registry.call("reveil").ok
    finally:
        surveillant.stop()

"""Les quatre plugins livrés, qui servent de documentation vivante.

S'ils cassent, l'exemple que l'on donne aux auteurs de plugins casse avec eux.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from capucine.core import plugin as contrat
from capucine.core.config import PROJECT_ROOT, Config
from capucine.core.registry import PluginRegistry

DOSSIER = PROJECT_ROOT / "plugins"


@pytest.fixture
def registre_livre(tmp_path: Path):
    annonces: list[str] = []
    contrat.set_announcer(annonces.append)
    registry = PluginRegistry(
        [DOSSIER],
        config=Config({"plugins": {"notes": {"lecture_par_defaut": 2}}}),
        data_root=tmp_path / "data",
    )
    registry.load_all()
    yield registry, annonces
    for nom in list(registry.plugins):
        registry.unload(nom, notify=False)
    contrat.set_announcer(None)


ENSEIGNANTS = {"heure", "minuteur", "notes", "systeme"}
ASSISTANCE = {"memoire", "recherche", "fichiers", "python", "projet", "documents"}
# Spécialisé sur UN projet, là où « projet » reste générique.
SPECIALISES = {"calculrisque"}
INTROSPECTION = {"apprentissage", "connaissances", "routines", "capacites"}


def test_tous_les_plugins_livres_se_chargent(registre_livre) -> None:
    registry, _ = registre_livre
    assert set(registry.plugins) == ENSEIGNANTS | ASSISTANCE | INTROSPECTION | SPECIALISES
    assert registry.failures() == []
    # Chacun expose au moins une compétence, et toutes ont un schéma d'outil.
    assert all(record.skills for record in registry.plugins.values())
    assert len(registry.tool_schemas()) == len(registry.skills)


# --- heure : lecture simple, parole et journal dissociés --------------------

def test_l_heure_est_dite_en_lettres_et_journalisee_en_chiffres(registre_livre) -> None:
    registry, _ = registre_livre
    resultat = registry.call("heure")
    assert resultat.ok
    assert resultat.speak.startswith("Il est ")
    assert any(c.isdigit() for c in resultat.display)
    # Ce qui est dit ne contient pas de chiffres : Piper les prononce mal.
    assert not any(c.isdigit() for c in resultat.speak)


def test_la_date_accepte_hier_aujourd_hui_demain(registre_livre) -> None:
    registry, _ = registre_livre
    for jour in ("hier", "aujourd'hui", "demain"):
        assert registry.call("date", {"jour": jour}).ok
    # L'énumération du schéma interdit au modèle d'inventer autre chose.
    refus = registry.call("date", {"jour": "après-demain"})
    assert not refus.ok and "doit valoir" in refus.speak


# --- minuteur : état de fond, annonce, nettoyage ---------------------------

def test_un_minuteur_annonce_a_echeance(registre_livre) -> None:
    registry, annonces = registre_livre
    assert "C'est parti" in registry.call("minuteur", {"secondes": 1, "libelle": "thé"}).speak
    assert "thé" in registry.call("minuteurs").speak

    limite = time.monotonic() + 5
    while not annonces and time.monotonic() < limite:
        time.sleep(0.05)
    assert annonces == ["C'est l'heure : thé."]
    assert registry.call("minuteurs").speak == "Aucun minuteur en cours."


def test_un_minuteur_sans_duree_le_dit(registre_livre) -> None:
    registry, _ = registre_livre
    assert "durée" in registry.call("minuteur").speak


def test_les_minuteurs_sont_annules_au_dechargement(registre_livre) -> None:
    # Sans on_unload(), chaque rechargement à chaud laisserait des minuteries
    # orphelines qui sonneraient dans le vide.
    registry, annonces = registre_livre
    registry.call("minuteur", {"secondes": 1, "libelle": "fantôme"})
    registry.unload("minuteur")
    time.sleep(1.3)
    assert annonces == []


def test_on_peut_annuler_un_minuteur_par_son_nom(registre_livre) -> None:
    registry, _ = registre_livre
    registry.call("minuteur", {"minutes": 5, "libelle": "pâtes"})
    registry.call("minuteur", {"minutes": 5, "libelle": "four"})
    assert "pâtes annulé" in registry.call("annuler_minuteur", {"libelle": "pâtes"}).speak
    assert "four" in registry.call("minuteurs").speak
    assert "annulé" in registry.call("annuler_minuteur").speak


# --- notes : persistance et confirmation -----------------------------------

def test_les_notes_survivent_dans_le_dossier_du_plugin(registre_livre, tmp_path) -> None:
    registry, _ = registre_livre
    assert registry.call("noter", {"texte": "appeler le plombier"}).speak == "C'est noté."
    registry.call("noter", {"texte": "acheter du pain"})

    fichier = tmp_path / "data" / "notes" / "notes.jsonl"
    assert fichier.exists()
    assert len(fichier.read_text(encoding="utf-8").strip().splitlines()) == 2

    # lecture_par_defaut = 2 vient de la configuration, pas du module.
    relecture = registry.call("mes_notes")
    assert "plombier" in relecture.speak and "pain" in relecture.speak


def test_on_retrouve_une_note_par_un_mot(registre_livre) -> None:
    registry, _ = registre_livre
    registry.call("noter", {"texte": "arroser les plantes"})
    assert "plantes" in registry.call("chercher_note", {"mot": "arroser"}).speak
    assert "Rien sur" in registry.call("chercher_note", {"mot": "licorne"}).speak


def test_effacer_les_notes_demande_confirmation(registre_livre) -> None:
    registry, _ = registre_livre
    registry.call("noter", {"texte": "une note"})

    demande = registry.call("effacer_notes")
    assert demande.needs_confirmation
    assert demande.speak == "Voulez-vous vraiment effacer toutes vos notes ?"
    assert "une note" in registry.call("mes_notes").speak   # rien n'a été effacé

    assert "effacée" in registry.call("effacer_notes", confirmed=True).speak
    assert registry.call("mes_notes").speak == "Vous n'avez aucune note."


def test_une_note_vide_est_refusee(registre_livre) -> None:
    registry, _ = registre_livre
    assert "à noter" in registry.call("noter", {"texte": "   "}).speak


# --- systeme : accès machine, dégradation propre ---------------------------

def test_l_etat_systeme_repond_toujours_quelque_chose(registre_livre) -> None:
    registry, _ = registre_livre
    resultat = registry.call("etat_systeme")
    assert resultat.ok
    assert "gigaoctets" in resultat.speak
    assert "disque_libre" in resultat.display


def test_le_volume_repond_meme_sans_carte_son(registre_livre) -> None:
    # En intégration continue il n'y a ni pactl ni carte son : la compétence
    # doit le dire, pas lever.
    registry, _ = registre_livre
    resultat = registry.call("volume")
    assert resultat.ok
    assert "volume" in resultat.display.lower()


def test_le_volume_borne_les_valeurs_extremes(registre_livre) -> None:
    registry, _ = registre_livre
    assert registry.call("volume", {"niveau": 500}).ok
    assert registry.call("volume", {"niveau": -50}).ok


def test_aucun_plugin_livre_n_ecrit_hors_de_son_dossier(registre_livre, tmp_path) -> None:
    registry, _ = registre_livre
    registry.call("noter", {"texte": "test"})
    ecrits = {p.parent.name for p in (tmp_path / "data").rglob("*") if p.is_file()}
    assert ecrits <= {"notes"}

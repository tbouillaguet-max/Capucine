"""L'application de bureau : le pont, les pièces, et le lanceur Windows.

Ce qui est éprouvé ici tourne **sans écran** : Qt sait rendre hors écran, et
les widgets se construisent, se peuplent et se mesurent exactement comme
devant un utilisateur. Ce qui n'est PAS éprouvé, et qu'il faut dire : le
comportement d'un vrai ``.exe`` sous Windows. Il se construit sur la machine
cible, et seul un essai là-bas le confirme.

Le lanceur, lui, est du Python ordinaire : sa logique de recherche — les
réglages, le dossier, les parents, l'interpréteur — s'éprouve partout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="l'interface de bureau réclame PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from lily.bureau import theme  # noqa: E402
from lily.bureau.widgets.conversation import Bulle, FilDeConversation, Systeme  # noqa: E402
from lily.bureau.widgets.etat import (  # noqa: E402
    DEPUIS_LE_PIPELINE,
    Activite,
    IndicateurDEtat,
)
from lily.bureau.widgets.fichiers import PanneauDesFichiers, _sans_ecraser, _taille  # noqa: E402
from lily.core.pipeline import State  # noqa: E402


@pytest.fixture(scope="module")
def application():
    """Une QApplication pour tout le fichier : Qt n'en veut qu'une."""
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(theme.feuille_de_style())
    yield app


# --- l'indicateur d'état ----------------------------------------------------

def test_chaque_etat_du_pipeline_a_son_activite(application) -> None:
    """Si un état apparaît dans le pipeline sans traduction ici, l'indicateur
    retomberait silencieusement sur « prête » — et mentirait."""
    for etat in State:
        assert etat.value in DEPUIS_LE_PIPELINE, f"« {etat.value} » n'est pas traduit"


def test_l_indicateur_suit_le_pipeline(application) -> None:
    indicateur = IndicateurDEtat()
    indicateur.depuis_le_pipeline(State.THINK.value)
    assert indicateur.pastille.activite is Activite.REFLECHIT
    indicateur.depuis_le_pipeline(State.SPEAK.value)
    assert indicateur.pastille.activite is Activite.PARLE
    assert indicateur.libelle.text() == Activite.PARLE.value


def test_seuls_les_etats_qui_avancent_sont_animes(application) -> None:
    """Une pastille immobile ne doit pas consommer un rafraîchissement toutes
    les quarante millisecondes pour dessiner la même chose."""
    indicateur = IndicateurDEtat()
    indicateur.regler(Activite.REFLECHIT)
    assert indicateur.pastille._minuteur.isActive()
    indicateur.regler(Activite.REPOS)
    assert not indicateur.pastille._minuteur.isActive()
    indicateur.regler(Activite.ETEINTE)
    assert not indicateur.pastille._minuteur.isActive()


def test_la_pastille_se_dessine_dans_tous_les_etats(application) -> None:
    """Un `paintEvent` qui lève laisse un trou noir dans la fenêtre."""
    indicateur = IndicateurDEtat()
    for activite in Activite:
        indicateur.regler(activite)
        indicateur.pastille._phase = 0.4
        image = indicateur.pastille.grab()
        assert not image.isNull()


# --- le fil de conversation -------------------------------------------------

def test_une_bulle_s_elargit_avec_son_texte(application) -> None:
    """Un QLabel qui renvoie à la ligne réclame le minimum : sans mesure, une
    phrase de trente mots tiendrait sur une colonne de deux cents pixels."""
    courte = Bulle("oui", de_lily=False)
    longue = Bulle("une phrase bien plus longue que la précédente, et qui doit "
                   "occuper davantage de largeur à l'écran", de_lily=True)
    assert courte.width() < longue.width()
    assert longue.width() <= Bulle.LARGEUR_MAX
    assert courte.width() >= Bulle.LARGEUR_MIN


def test_les_phrases_s_ajoutent_a_la_bulle_en_cours(application) -> None:
    fil = FilDeConversation()
    fil.phrase_de_lily("Première phrase.")
    fil.phrase_de_lily("Seconde phrase.")
    assert fil._bulle_ouverte is not None
    assert fil._bulle_ouverte.corps.text() == "Première phrase. Seconde phrase."


def test_un_tour_ecrit_remet_la_reponse_apres_la_question(application) -> None:
    fil = FilDeConversation()
    fil.dire_vous("quelle heure est-il")
    fil.phrase_de_lily("Il est midi.")
    fil.terminer_le_tour("quelle heure est-il", "Il est midi.", "sans modèle · heure")

    textes = _textes(fil)
    assert textes == ["quelle heure est-il", "Il est midi."]
    assert fil._bulle_ouverte is None


def test_un_tour_vocal_insere_votre_phrase_a_sa_place(application) -> None:
    """Au micro, la transcription n'est connue qu'à la fin du tour : sans
    insertion, votre question s'afficherait SOUS la réponse."""
    fil = FilDeConversation()
    fil.phrase_de_lily("Il est midi.")            # elle répond d'abord
    fil.terminer_le_tour("quelle heure est-il", "Il est midi.", "sans modèle")

    assert _textes(fil) == ["quelle heure est-il", "Il est midi."]


def test_la_legende_dit_d_ou_vient_la_reponse(application) -> None:
    fil = FilDeConversation()
    fil.dire_vous("note ça")
    fil.terminer_le_tour("note ça", "C'est noté.", "outil reconnu · noter · 12 ms")
    bulle = _dernieres_bulles(fil)[-1]
    assert bulle.legende.isVisible() or bulle.legende.text()
    assert "noter" in bulle.legende.text()


def test_une_ligne_systeme_prend_toute_la_largeur(application) -> None:
    fil = FilDeConversation()
    fil.resize(900, 400)
    fil.systeme("Aucune voix : les réponses seront affichées.", alerte=True)
    lignes = [fil._pile.itemAt(i).widget() for i in range(fil._pile.count() - 1)]
    systeme = [w for ligne in lignes if ligne for w in ligne.findChildren(Systeme)]
    assert systeme, "la ligne système n'a pas été posée"


def test_vider_le_fil_n_en_laisse_rien(application) -> None:
    fil = FilDeConversation()
    fil.dire_vous("bonjour")
    fil.phrase_de_lily("Bonjour.")
    fil.vider()
    assert fil._pile.count() == 1          # il ne reste que l'étirement
    assert fil._bulle_ouverte is None


# --- le panneau des fichiers ------------------------------------------------

def test_le_panneau_liste_le_dossier(application, tmp_path: Path) -> None:
    (tmp_path / "rapport.docx").write_bytes(b"x" * 2048)
    (tmp_path / "notes.md").write_text("bonjour", encoding="utf-8")

    panneau = PanneauDesFichiers()
    panneau.ouvrir_le_dossier(tmp_path)
    assert panneau.liste.count() == 2
    assert "2 éléments" in panneau.pied.text()


def test_les_fichiers_deposes_sont_copies_jamais_deplaces(
    application, tmp_path: Path
) -> None:
    """Si la copie tourne mal, l'original doit être resté là où il était."""
    source = tmp_path / "source"
    source.mkdir()
    original = source / "budget.xlsx"
    original.write_bytes(b"contenu")
    travail = tmp_path / "travail"

    panneau = PanneauDesFichiers()
    panneau.ouvrir_le_dossier(travail)
    copies = panneau.accueillir([original])

    assert len(copies) == 1
    assert copies[0].read_bytes() == b"contenu"
    assert original.exists(), "l'original a été déplacé"


def test_un_depot_n_ecrase_jamais_un_fichier_existant(
    application, tmp_path: Path
) -> None:
    travail = tmp_path / "travail"
    travail.mkdir()
    (travail / "notes.md").write_text("le premier", encoding="utf-8")
    autre = tmp_path / "notes.md"
    autre.write_text("le second", encoding="utf-8")

    panneau = PanneauDesFichiers()
    panneau.ouvrir_le_dossier(travail)
    (copie,) = panneau.accueillir([autre])

    assert copie.name == "notes (2).md"
    assert (travail / "notes.md").read_text(encoding="utf-8") == "le premier"


def test_ce_qui_change_est_signale(application, tmp_path: Path) -> None:
    """Un assistant qui écrit un fichier sans le dire oblige à aller vérifier
    dans l'explorateur."""
    (tmp_path / "ancien.txt").write_text("a", encoding="utf-8")
    panneau = PanneauDesFichiers()
    panneau.ouvrir_le_dossier(tmp_path)
    assert not panneau._nouveaux           # le premier regard ne signale rien

    (tmp_path / "produit_par_lily.py").write_text("print()", encoding="utf-8")
    panneau.rafraichir()
    assert "produit_par_lily.py" in panneau._nouveaux
    assert "nouveau" in panneau.pied.text()

    panneau.marquer_comme_vus()
    assert not panneau._nouveaux


def test_les_tailles_sont_lisibles() -> None:
    assert _taille(512) == "512 o"
    assert _taille(2048) == "2.0 ko"
    assert _taille(3 * 1024 * 1024) == "3.0 Mo"


def test_un_nom_libre_est_trouve(tmp_path: Path) -> None:
    cible = tmp_path / "a.txt"
    assert _sans_ecraser(cible) == cible
    cible.write_text("", encoding="utf-8")
    assert _sans_ecraser(cible).name == "a (2).txt"


# --- le lanceur Windows -----------------------------------------------------

def _fausse_installation(racine: Path) -> Path:
    racine.mkdir(parents=True, exist_ok=True)
    (racine / "lily_bureau.py").write_text("", encoding="utf-8")
    (racine / "lily").mkdir(exist_ok=True)
    return racine


def test_le_lanceur_reconnait_une_installation(tmp_path: Path) -> None:
    from deploy import lanceur

    assert not lanceur.ressemble_a_lily(tmp_path)
    _fausse_installation(tmp_path / "Lily")
    assert lanceur.ressemble_a_lily(tmp_path / "Lily")


def test_le_lanceur_se_trouve_depuis_un_sous_dossier(tmp_path: Path) -> None:
    """Le cas normal : l'exécutable est posé dans le dépôt, ou sous lui."""
    from deploy import lanceur

    depot = _fausse_installation(tmp_path / "Lily")
    profond = depot / "deploy" / "dist"
    profond.mkdir(parents=True)
    assert lanceur.trouver_lily(profond) == depot


def test_le_lanceur_retient_le_dossier_qu_on_lui_indique(tmp_path: Path) -> None:
    from deploy import lanceur

    depot = _fausse_installation(tmp_path / "ailleurs" / "Lily")
    a_cote = tmp_path / "bureau"
    a_cote.mkdir()

    assert lanceur.trouver_lily(a_cote) is None
    lanceur.ecrire_les_reglages(a_cote, depot)
    assert (a_cote / lanceur.FICHIER_DE_REGLAGES).is_file()
    assert lanceur.trouver_lily(a_cote) == depot


def test_un_reglage_qui_pointe_dans_le_vide_est_ignore(tmp_path: Path) -> None:
    """Un dépôt déplacé ne doit pas condamner le lanceur : il redemande."""
    from deploy import lanceur

    a_cote = tmp_path / "bureau"
    a_cote.mkdir()
    lanceur.ecrire_les_reglages(a_cote, tmp_path / "parti" / "ailleurs")
    assert lanceur.lire_les_reglages(a_cote) is None


def test_le_lanceur_trouve_l_environnement_python(tmp_path: Path) -> None:
    from deploy import lanceur

    depot = _fausse_installation(tmp_path / "Lily")
    assert lanceur.trouver_l_interpreteur(depot) is None

    scripts = depot / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")
    scripts.mkdir(parents=True)
    nom = "pythonw.exe" if sys.platform == "win32" else "python"
    (scripts / nom).write_text("", encoding="utf-8")
    trouve = lanceur.trouver_l_interpreteur(depot)
    assert trouve is not None and trouve.name == nom


def test_le_script_de_construction_refuse_ailleurs_que_sous_windows() -> None:
    """PyInstaller ne compile pas pour une autre plateforme que la sienne.
    Le dire avant de construire vaut mieux que de livrer un binaire Linux."""
    from deploy import construire_exe

    probleme = construire_exe.verifier()
    if sys.platform != "win32":
        assert probleme is not None
        assert "Windows" in probleme


# --- utilitaires du fichier -------------------------------------------------

def _dernieres_bulles(fil: FilDeConversation) -> list[Bulle]:
    bulles: list[Bulle] = []
    for index in range(fil._pile.count() - 1):
        ligne = fil._pile.itemAt(index).widget()
        if ligne is not None:
            bulles.extend(ligne.findChildren(Bulle))
    return bulles


def _textes(fil: FilDeConversation) -> list[str]:
    return [bulle.corps.text() for bulle in _dernieres_bulles(fil)]

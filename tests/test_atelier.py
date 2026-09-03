"""L'atelier : la frontière entre Lily et vos fichiers.

C'est le fichier de tests le plus important de cette capacité. La commande
arrive par la voix, la transcription est imparfaite, le modèle choisit
parfois mal ses arguments : ce qui protège vraiment, c'est ce qui suit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lily.core.atelier import Atelier, AtelierError, depuis_config
from lily.core.config import Config
from lily.core.errors import SkillRefused


@pytest.fixture
def espace(tmp_path: Path) -> Atelier:
    (tmp_path / "projet").mkdir()
    (tmp_path / "projet" / "notes.txt").write_text("bonjour", encoding="utf-8")
    (tmp_path / "projet" / ".env").write_text("CLE=secret", encoding="utf-8")
    (tmp_path / "dehors.txt").write_text("interdit", encoding="utf-8")
    return Atelier(racines=[tmp_path / "projet"], corbeille=tmp_path / "corbeille")


def test_un_atelier_ferme_refuse_tout() -> None:
    # La capacité est livrée inerte : rien n'est accessible avant décision.
    ferme = Atelier()
    assert not ferme.ouvert
    with pytest.raises(AtelierError, match="atelier.racines"):
        ferme.resoudre("n_importe_quoi.txt")


def test_un_refus_est_une_reponse_pas_un_plantage(espace: Atelier) -> None:
    # AtelierError hérite de SkillRefused : le registre le prononce tel quel
    # au lieu de le traduire en « je n'ai pas pu exécuter cette commande ».
    with pytest.raises(SkillRefused):
        espace.resoudre("/etc/passwd")


@pytest.mark.parametrize(
    "tentative",
    ["../dehors.txt", "/etc/passwd", "../../etc/hosts", "~/.ssh/id_rsa"],
)
def test_on_ne_sort_pas_de_l_atelier(espace: Atelier, tentative: str) -> None:
    with pytest.raises(AtelierError):
        espace.resoudre(tentative)


def test_un_lien_symbolique_ne_sert_pas_de_passage(espace: Atelier, tmp_path: Path) -> None:
    # `resolve()` suit les liens : poser un lien dans l'atelier ne donne pas
    # accès à sa cible.
    lien = tmp_path / "projet" / "raccourci.txt"
    try:
        lien.symlink_to(tmp_path / "dehors.txt")
    except OSError:  # pragma: no cover - systèmes sans liens symboliques
        pytest.skip("liens symboliques indisponibles")
    with pytest.raises(AtelierError, match="hors de l'atelier"):
        espace.resoudre("raccourci.txt")


@pytest.mark.parametrize("nom", [".env", "cle.pem", "id_rsa", "mes_credentials.json"])
def test_les_secrets_restent_hors_de_portee(espace: Atelier, tmp_path: Path, nom: str) -> None:
    # Même à l'intérieur d'une racine autorisée.
    (tmp_path / "projet" / nom).write_text("x", encoding="utf-8")
    with pytest.raises(AtelierError, match="protégé"):
        espace.resoudre(nom)


def test_les_dossiers_sensibles_sont_refuses(tmp_path: Path) -> None:
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "notes.txt").write_text("x", encoding="utf-8")
    espace = Atelier(racines=[tmp_path])
    with pytest.raises(AtelierError, match="sensible"):
        espace.resoudre(".ssh/notes.txt")


def test_un_chemin_relatif_part_de_la_premiere_racine(espace: Atelier) -> None:
    assert espace.lire("notes.txt") == "bonjour"


def test_toute_reecriture_laisse_une_sauvegarde(espace: Atelier, tmp_path: Path) -> None:
    espace.ecrire("notes.txt", "bonsoir")
    assert espace.lire("notes.txt") == "bonsoir"
    sauvegardes = list((tmp_path / "projet").glob("notes.txt.*.sauvegarde"))
    assert len(sauvegardes) == 1
    assert sauvegardes[0].read_text(encoding="utf-8") == "bonjour"


def test_completer_un_fichier_ne_le_sauvegarde_pas(espace: Atelier, tmp_path: Path) -> None:
    espace.ecrire("notes.txt", " et bonsoir", ajouter=True)
    assert espace.lire("notes.txt") == "bonjour et bonsoir"
    assert not list((tmp_path / "projet").glob("*.sauvegarde"))


def test_rien_n_est_supprime_tout_va_a_la_corbeille(espace: Atelier, tmp_path: Path) -> None:
    # Un assistant vocal ne doit rien détruire d'irrécupérable.
    destination = espace.jeter("notes.txt")
    assert not (tmp_path / "projet" / "notes.txt").exists()
    assert destination.read_text(encoding="utf-8") == "bonjour"
    assert destination.parent == tmp_path / "corbeille"


def test_le_mode_lecture_seule_bloque_toute_ecriture(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("bonjour", encoding="utf-8")
    espace = Atelier(racines=[tmp_path], lecture_seule=True)
    assert espace.lire("notes.txt") == "bonjour"
    with pytest.raises(AtelierError, match="lecture seule"):
        espace.ecrire("notes.txt", "bonsoir")
    with pytest.raises(AtelierError, match="lecture seule"):
        espace.jeter("notes.txt")


def test_un_fichier_trop_gros_n_est_pas_lu_d_un_bloc(tmp_path: Path) -> None:
    (tmp_path / "gros.txt").write_text("x" * 200_000, encoding="utf-8")
    espace = Atelier(racines=[tmp_path], taille_max_ko=10)
    with pytest.raises(AtelierError, match="taille_max_ko"):
        espace.lire("gros.txt")


def test_lister_masque_les_fichiers_proteges(espace: Atelier) -> None:
    noms = {chemin.name for chemin in espace.lister(".")}
    assert "notes.txt" in noms
    assert ".env" not in noms


def test_deplacer_reste_dans_l_atelier(espace: Atelier, tmp_path: Path) -> None:
    (tmp_path / "projet" / "archives").mkdir()
    arrivee = espace.deplacer("notes.txt", "archives/notes.txt")
    assert arrivee.read_text(encoding="utf-8") == "bonjour"
    with pytest.raises(AtelierError):
        espace.deplacer("archives/notes.txt", "../evasion.txt")


# --- construction depuis la configuration -----------------------------------

def test_la_configuration_par_defaut_ouvre_un_atelier_vide() -> None:
    from lily.core.config import load_config

    espace = depuis_config(load_config(profile="pc", environ={}))
    # Le défaut livré est volontairement inerte.
    assert not espace.ouvert


def test_une_racine_inexistante_est_ignoree_pas_fatale(tmp_path: Path) -> None:
    config = Config({"atelier": {"racines": [str(tmp_path), "/dossier/qui/n/existe/pas"]}})
    espace = depuis_config(config)
    assert espace.racines == [tmp_path.resolve()]


def test_les_reglages_sont_lus(tmp_path: Path) -> None:
    config = Config({"atelier": {
        "racines": [str(tmp_path)], "lecture_seule": True, "taille_max_ko": 42,
        "corbeille": str(tmp_path / "poubelle"),
    }})
    espace = depuis_config(config)
    assert espace.lecture_seule and espace.taille_max_ko == 42
    assert espace.corbeille == tmp_path / "poubelle"


# --- ce que l'utilisateur voit au démarrage ---------------------------------

def test_une_racine_introuvable_est_retenue_pour_etre_dite(tmp_path: Path) -> None:
    # Un atelier vide sans explication, c'est dix minutes perdues à se demander
    # pourquoi toutes les compétences refusent.
    config = Config({"atelier": {"racines": [str(tmp_path / "absent")]}})
    espace = depuis_config(config)
    assert not espace.ouvert
    assert espace.racines_ignorees == [str(tmp_path / "absent")]


def test_un_atelier_bien_configure_ne_signale_rien(tmp_path: Path) -> None:
    espace = depuis_config(Config({"atelier": {"racines": [str(tmp_path)]}}))
    assert espace.ouvert and espace.racines_ignorees == []


def test_les_motifs_proteges_tolerent_les_separateurs_windows(tmp_path: Path) -> None:
    # Sous Windows un chemin s'écrit avec des antislashs ; un motif comme
    # « .git/config » ne le rencontrerait jamais sans normalisation.
    espace = Atelier(racines=[tmp_path], motifs_interdits=(".git/config",))
    faux_chemin = Path(str(tmp_path) + "/.git/config")
    with pytest.raises(AtelierError, match="protégé"):
        espace._verifier_interdits(faux_chemin)


# --- correctif A : ne pas détruire un fichier binaire ------------------------

def test_on_n_ecrit_pas_de_texte_dans_un_fichier_binaire(tmp_path: Path) -> None:
    # L'atelier écrit en UTF-8. Un .xlsx réécrit ainsi n'est pas modifié :
    # il est détruit.
    (tmp_path / "budget.xlsx").write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    espace = Atelier(racines=[tmp_path])
    with pytest.raises(AtelierError, match="n'est pas un fichier texte"):
        espace.ecrire("budget.xlsx", "du texte")
    # Le fichier est intact.
    assert (tmp_path / "budget.xlsx").read_bytes().startswith(b"PK\x03\x04")


def test_le_verdict_porte_sur_le_contenu_pas_sur_l_extension(tmp_path: Path) -> None:
    # Une extension ment souvent : c'est le contenu qui tranche quand le
    # fichier existe.
    (tmp_path / "piege.txt").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    (tmp_path / "vraiment.xlsx").write_text("en fait du texte", encoding="utf-8")
    espace = Atelier(racines=[tmp_path])

    with pytest.raises(AtelierError, match="n'est pas un fichier texte"):
        espace.ecrire("piege.txt", "bonjour")
    # Celui-ci est réellement du texte malgré son extension : on l'accepte.
    espace.ecrire("vraiment.xlsx", "toujours du texte")


def test_on_ne_cree_pas_un_document_office_en_texte(tmp_path: Path) -> None:
    espace = Atelier(racines=[tmp_path])
    for nom in ("rapport.docx", "budget.xlsx", "presentation.pptx", "note.pdf"):
        with pytest.raises(AtelierError, match="pas du texte"):
            espace.ecrire(nom, "du texte")


def test_les_fichiers_texte_restent_ecrivables(tmp_path: Path) -> None:
    espace = Atelier(racines=[tmp_path])
    for nom in ("script.py", "notes.md", "donnees.csv", "config.toml", "sans_extension"):
        espace.ecrire(nom, "contenu\n")
        assert espace.lire(nom) == "contenu\n"


def test_un_accent_coupe_en_fin_d_extrait_n_est_pas_du_binaire(tmp_path: Path) -> None:
    # Le test de binarité lit les 8 premiers kilo-octets : la coupure peut
    # tomber au milieu d'un caractère UTF-8 multi-octets.
    (tmp_path / "long.txt").write_text("é" * 9000, encoding="utf-8")
    espace = Atelier(racines=[tmp_path])
    espace.ecrire("long.txt", "remplacé")   # ne doit pas lever
    assert espace.lire("long.txt") == "remplacé"


def test_un_fichier_vide_est_ecrivable(tmp_path: Path) -> None:
    (tmp_path / "vide.txt").write_bytes(b"")
    espace = Atelier(racines=[tmp_path])
    espace.ecrire("vide.txt", "enfin du contenu")

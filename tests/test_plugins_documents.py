"""Word, Excel, PowerPoint, PDF : ce que l'UTF-8 ne sait pas lire."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lily.core import plugin as contrat
from lily.core.atelier import Atelier
from lily.core.config import PROJECT_ROOT, Config
from lily.core.registry import PluginRegistry

DOSSIER = PROJECT_ROOT / "plugins"


@pytest.fixture
def bureau(tmp_path: Path):
    """Un atelier contenant de vrais documents Office."""
    travail = tmp_path / "bureau"
    travail.mkdir()
    contrat.set_atelier(Atelier(racines=[travail]))
    contrat.set_model_access(lambda p, **k: "Un résumé du modèle.")

    def _registre(config: dict | None = None) -> PluginRegistry:
        registry = PluginRegistry(
            [DOSSIER], config=Config(config or {}), data_root=tmp_path / "data"
        )
        registry.load_all()
        return registry

    yield {"travail": travail, "registre": _registre}
    contrat.set_atelier(None)
    contrat.set_model_access(None)


def _word(chemin: Path) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Rapport trimestriel de valorisation.")
    document.add_paragraph("Le multiple médian ressort à 12,4x.")
    tableau = document.add_table(rows=2, cols=2)
    tableau.rows[0].cells[0].text = "Société"
    tableau.rows[1].cells[0].text = "Exemple SA"
    document.save(str(chemin))


def _excel(chemin: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    classeur = openpyxl.Workbook()
    feuille = classeur.active
    feuille.title = "Synthèse"
    feuille.append(["Titre", "Valeur"])
    feuille.append(["Multiple médian", 12.4])
    classeur.create_sheet("Détail").append(["a", "b"])
    classeur.save(str(chemin))


def _powerpoint(chemin: Path) -> None:
    pptx = pytest.importorskip("pptx")
    presentation = pptx.Presentation()
    diapositive = presentation.slides.add_slide(presentation.slide_layouts[5])
    diapositive.shapes.title.text = "Valorisation 2026"
    diapositive.notes_slide.notes_text_frame.text = "Insister sur l'écart de multiple."
    presentation.save(str(chemin))


# --- lecture ----------------------------------------------------------------

def test_un_word_est_lu_avec_ses_tableaux(bureau) -> None:
    _word(bureau["travail"] / "rapport.docx")
    resultat = bureau["registre"]().call("lire_document", {"chemin": "rapport.docx"})
    assert "multiple médian" in resultat.display
    # Un tableau porte souvent l'essentiel : il ne doit pas être sauté.
    assert "Exemple SA" in resultat.display


def test_un_excel_rend_ses_valeurs_pas_ses_formules(bureau) -> None:
    _excel(bureau["travail"] / "budget.xlsx")
    registry = bureau["registre"]()
    resultat = registry.call("lire_tableur", {"chemin": "budget.xlsx", "feuille": "Synthèse"})
    assert "12.4" in resultat.display
    assert "=" not in resultat.display


def test_on_liste_les_feuilles_d_un_classeur(bureau) -> None:
    _excel(bureau["travail"] / "budget.xlsx")
    resultat = bureau["registre"]().call("feuilles_du_tableur", {"chemin": "budget.xlsx"})
    assert "Synthèse" in resultat.speak and "Détail" in resultat.speak
    assert "ligne" in resultat.display


def test_une_feuille_inconnue_liste_celles_qui_existent(bureau) -> None:
    _excel(bureau["travail"] / "budget.xlsx")
    resultat = bureau["registre"]().call(
        "lire_tableur", {"chemin": "budget.xlsx", "feuille": "inexistante"}
    )
    assert "Synthèse" in resultat.speak


def test_un_powerpoint_rend_ses_diapositives_et_ses_notes(bureau) -> None:
    _powerpoint(bureau["travail"] / "presentation.pptx")
    resultat = bureau["registre"]().call("lire_document", {"chemin": "presentation.pptx"})
    assert "Valorisation 2026" in resultat.display
    # Les notes du présentateur disent souvent ce que la diapositive tait.
    assert "écart de multiple" in resultat.display


def test_le_resume_passe_par_le_modele(bureau) -> None:
    _word(bureau["travail"] / "rapport.docx")
    resultat = bureau["registre"]().call("resumer_document", {"chemin": "rapport.docx"})
    assert resultat.speak == "Un résumé du modèle."


def test_on_cherche_a_travers_les_documents(bureau) -> None:
    _word(bureau["travail"] / "rapport.docx")
    _excel(bureau["travail"] / "budget.xlsx")
    registry = bureau["registre"]()

    trouvaille = registry.call("chercher_dans_documents", {"texte": "médian"})
    assert "rapport.docx" in trouvaille.display and "budget.xlsx" in trouvaille.display
    assert "Rien sur" in registry.call("chercher_dans_documents", {"texte": "licorne"}).speak


# --- refus ------------------------------------------------------------------

def test_l_ancien_format_binaire_est_refuse_avec_le_remede(bureau) -> None:
    (bureau["travail"] / "vieux.doc").write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 32)
    resultat = bureau["registre"]().call("lire_document", {"chemin": "vieux.doc"})
    assert "ancien format" in resultat.speak
    assert ".docx" in resultat.speak


def test_un_format_inconnu_est_refuse(bureau) -> None:
    (bureau["travail"] / "archive.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 32)
    resultat = bureau["registre"]().call("lire_document", {"chemin": "archive.zip"})
    assert "ne sais pas ouvrir" in resultat.speak


def test_un_document_hors_de_l_atelier_est_refuse(bureau) -> None:
    resultat = bureau["registre"]().call("lire_document", {"chemin": "/etc/passwd"})
    assert "hors de l'atelier" in resultat.speak


def test_un_document_trop_gros_est_refuse(bureau) -> None:
    _word(bureau["travail"] / "rapport.docx")
    registry = bureau["registre"]({"plugins": {"documents": {"taille_max_mo": 0.000001}}})
    resultat = registry.call("lire_document", {"chemin": "rapport.docx"})
    assert "au-delà de la limite" in resultat.speak


def test_une_bibliotheque_absente_nomme_le_paquet(bureau, monkeypatch) -> None:
    # Chaque format est importé séparément : l'absence d'openpyxl ne doit pas
    # empêcher de lire un .docx. On fabrique les fichiers AVANT de simuler
    # l'absence, sinon c'est le test lui-même qui ne peut plus les créer.
    _excel(bureau["travail"] / "budget.xlsx")
    _word(bureau["travail"] / "rapport.docx")
    registry = bureau["registre"]()

    monkeypatch.delitem(sys.modules, "openpyxl", raising=False)
    monkeypatch.setattr("builtins.__import__", _import_sans("openpyxl"))

    refus = registry.call("lire_document", {"chemin": "budget.xlsx"})
    assert "openpyxl" in refus.speak and "pip install" in refus.speak
    # Le .docx, lui, reste lisible.
    assert "valorisation" in registry.call(
        "lire_document", {"chemin": "rapport.docx"}
    ).display.lower()


def _import_sans(interdit: str):
    import builtins

    vrai = builtins.__import__

    def _import(nom, *args, **kwargs):
        if nom == interdit:
            raise ImportError(f"pas de module {interdit}")
        return vrai(nom, *args, **kwargs)

    return _import


def test_un_document_sans_texte_le_dit(bureau) -> None:
    docx = pytest.importorskip("docx")
    docx.Document().save(str(bureau["travail"] / "vide.docx"))
    resultat = bureau["registre"]().call("lire_document", {"chemin": "vide.docx"})
    assert "ne contient pas de texte" in resultat.speak

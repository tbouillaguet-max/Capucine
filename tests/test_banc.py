"""Le banc de mesure : il doit noter juste, et avec le bon interpréteur."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.banc_de_code import Tache, interpreteur_du_projet, noter


@pytest.fixture
def tache() -> Tache:
    return Tache(
        nom="essai", description="peu importe",
        verification="assert somme(2, 3) == 5\n", famille="logique",
    )


# --- la notation -------------------------------------------------------------

def test_du_bon_code_passe(tache: Tache, tmp_path: Path) -> None:
    reussi, raison = noter("def somme(a, b):\n    return a + b\n", tache, tmp_path, 30.0)
    assert reussi and "passée" in raison


def test_du_code_faux_mais_qui_tourne_echoue(tache: Tache, tmp_path: Path) -> None:
    # Le banc note en EXÉCUTANT : un résultat faux ne passe pas parce qu'il
    # compile.
    reussi, raison = noter("def somme(a, b):\n    return a * b\n", tache, tmp_path, 30.0)
    assert not reussi and "AssertionError" in raison


def test_une_api_inventee_echoue_avant_meme_de_tourner(tmp_path: Path) -> None:
    exigeante = Tache(
        nom="api", description="x", verification="assert True\n",
        famille="api", doit_appeler=["capped_weights"],
    )
    reussi, raison = noter("def f():\n    return 1\n", exigeante, tmp_path, 30.0)
    assert not reussi and "n'appelle pas capped_weights" in raison


def test_un_code_vide_ne_passe_pas(tache: Tache, tmp_path: Path) -> None:
    assert noter("   \n", tache, tmp_path, 30.0) == (False, "aucun code produit")


# --- l'interpréteur ----------------------------------------------------------

def _faux_venv(racine: Path, nom: str, relatif: str) -> Path:
    dossier = racine / nom
    (dossier / Path(relatif).parent).mkdir(parents=True)
    (dossier / "pyvenv.cfg").write_text("home = /usr", encoding="utf-8")
    python = dossier / relatif
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    return python


def test_l_interpreteur_du_projet_est_prefere(tmp_path: Path) -> None:
    """Le défaut qui notait en échec un code parfaitement correct.

    Le modèle avait écrit le bon import et le bon appel ; la vérification
    tournait avec le Python de Capucine, sans pandas, et rendait
    « ModuleNotFoundError ». Le banc mesurait la coïncidence de deux
    installations, pas le modèle.
    """
    attendu = _faux_venv(tmp_path, "env311", "bin/python3")
    assert interpreteur_du_projet(tmp_path) == str(attendu)


def test_un_venv_windows_est_reconnu(tmp_path: Path) -> None:
    attendu = _faux_venv(tmp_path, ".venv", "Scripts/python.exe")
    assert interpreteur_du_projet(tmp_path) == str(attendu)


def test_sans_environnement_on_garde_celui_de_capucine(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("", encoding="utf-8")
    assert interpreteur_du_projet(tmp_path) == sys.executable


def test_un_interpreteur_impose_gagne(tmp_path: Path) -> None:
    _faux_venv(tmp_path, "env311", "bin/python3")
    assert interpreteur_du_projet(tmp_path, Path("/usr/bin/python3")) == "/usr/bin/python3"


def test_un_dossier_sans_pyvenv_n_est_pas_un_environnement(tmp_path: Path) -> None:
    # Un dossier qui ressemble à un venv mais n'en est pas un ne doit pas
    # fournir l'interpréteur.
    (tmp_path / "faux" / "bin").mkdir(parents=True)
    (tmp_path / "faux" / "bin" / "python3").write_text("", encoding="utf-8")
    assert interpreteur_du_projet(tmp_path) == sys.executable

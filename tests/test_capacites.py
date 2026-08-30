"""Lister ses capacités liste le dossier plugins/, pas l'atelier de l'utilisateur.

Avant cette compétence, « liste tes capacités » retombait sur
``chercher_dans_fichiers`` (une compétence de l'atelier) et refusait tant
qu'aucun dossier de travail n'était ouvert — alors que la question ne portait
sur aucun dossier de travail.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from capucine.core.config import PROJECT_ROOT
from capucine.core.journal import JournalDesAppels
from capucine.core.registry import PluginRegistry
from capucine.core import plugin as contrat

PLUGINS_LIVRES = PROJECT_ROOT / "plugins"


@pytest.fixture
def banc(tmp_path: Path):
    dossier = tmp_path / "plugins"
    dossier.mkdir()
    for nom in ("capacites.py", "heure.py"):
        shutil.copy2(PLUGINS_LIVRES / nom, dossier / nom)

    registry = PluginRegistry([dossier], data_root=tmp_path / "data")
    contrat.set_registre(registry)
    contrat.set_dossier_des_plugins(dossier)
    contrat.set_journal(JournalDesAppels())
    registry.load_all()
    yield {"dossier": dossier, "registre": registry}
    for nom in list(registry.plugins):
        registry.unload(nom, notify=False)
    contrat.set_registre(None)
    contrat.set_dossier_des_plugins(None)
    contrat.set_journal(None)


def test_elle_liste_les_fichiers_du_dossier_plugins(banc) -> None:
    resultat = banc["registre"].call("lister_mes_capacites")
    assert resultat.ok
    assert "capacites.py" in resultat.display
    assert "heure.py" in resultat.display
    assert "2 fichiers de capacités" in resultat.speak


def test_elle_ne_depend_pas_d_un_atelier_ouvert(banc) -> None:
    # Aucun atelier configuré ici : ça ne doit rien changer, la compétence ne
    # touche pas aux fichiers de l'utilisateur.
    resultat = banc["registre"].call("lister_mes_capacites")
    assert resultat.ok


def test_dossier_vide(tmp_path: Path) -> None:
    dossier = tmp_path / "plugins"
    dossier.mkdir()
    shutil.copy2(PLUGINS_LIVRES / "capacites.py", dossier / "capacites.py")

    registry = PluginRegistry([dossier], data_root=tmp_path / "data")
    contrat.set_registre(registry)
    contrat.set_dossier_des_plugins(dossier)
    registry.load_all()
    try:
        (dossier / "capacites.py").unlink()
        resultat = registry.call("lister_mes_capacites")
        assert resultat.ok
        assert "aucun fichier" in resultat.speak.lower()
    finally:
        for nom in list(registry.plugins):
            registry.unload(nom, notify=False)
        contrat.set_registre(None)
        contrat.set_dossier_des_plugins(None)

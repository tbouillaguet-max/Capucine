"""Fixtures partagées.

Aucun test ne télécharge de modèle, n'ouvre de micro ni ne contacte de service :
la boucle de plugins doit être prouvable sur une machine nue.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from lily.core.config import Config  # noqa: E402
from lily.core.registry import PluginRegistry  # noqa: E402


@pytest.fixture
def dossier_plugins(tmp_path: Path) -> Path:
    dossier = tmp_path / "plugins"
    dossier.mkdir()
    return dossier


@pytest.fixture
def ecrire_plugin(dossier_plugins: Path):
    """Écrit un fichier de plugin et retourne son chemin."""

    def _ecrire(nom: str, code: str) -> Path:
        chemin = dossier_plugins / nom
        chemin.write_text(code, encoding="utf-8")
        return chemin

    return _ecrire


@pytest.fixture
def registre(dossier_plugins: Path, tmp_path: Path):
    """Fabrique de registres, avec nettoyage des modules chargés."""
    cree: list[PluginRegistry] = []

    def _registre(config: Config | None = None, **kwargs) -> PluginRegistry:
        registry = PluginRegistry(
            [dossier_plugins],
            config=config,
            data_root=tmp_path / "data",
            **kwargs,
        )
        cree.append(registry)
        return registry

    yield _registre

    for registry in cree:
        for nom in list(registry.plugins):
            registry.unload(nom, notify=False)

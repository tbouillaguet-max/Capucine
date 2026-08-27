"""API publique des plugins.

C'est **le seul module** qu'un plugin doit importer :

    from capucine.plugin import skill, get_config, data_dir, announce

Ce fichier est une façade délibérément mince sur ``capucine.core.plugin``.
Le cœur peut être réorganisé sans qu'aucun plugin n'ait à changer une ligne.
"""

from __future__ import annotations

from .core.plugin import (
    SkillSpec,
    announce,
    data_dir,
    get_config,
    skill,
)
from .core.plugin import get_logger_for_plugin as get_logger

__all__ = [
    "skill",
    "get_config",
    "get_logger",
    "data_dir",
    "announce",
    "SkillSpec",
]

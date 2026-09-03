"""API publique des plugins.

C'est **le seul module** qu'un plugin doit importer :

    from lily.plugin import skill, get_config, data_dir, announce

Ce fichier est une façade délibérément mince sur ``lily.core.plugin``.
Le cœur peut être réorganisé sans qu'aucun plugin n'ait à changer une ligne.
"""

from __future__ import annotations

from .core.errors import SkillRefused
from .core.plugin import (
    SkillSpec,
    announce,
    appeler_competence,
    apprentissage,
    atelier,
    catalogue,
    connaissances,
    conversation,
    corpus,
    data_dir,
    demander_au_modele,
    dossier_des_plugins,
    get_config,
    journal,
    memoire,
    skill,
)
from .core.plugin import get_logger_for_plugin as get_logger

__all__ = [
    "skill",
    "get_config",
    "get_logger",
    "data_dir",
    "announce",
    "demander_au_modele",
    "atelier",
    "catalogue",
    "memoire",
    "conversation",
    "apprentissage",
    "connaissances",
    "corpus",
    "journal",
    "appeler_competence",
    "dossier_des_plugins",
    "SkillSpec",
    "SkillRefused",
]

"""Exceptions du cœur.

Règle générale : aucune de ces exceptions ne doit remonter jusqu'à la boucle
principale. Le pipeline les convertit en phrase parlée.
"""

from __future__ import annotations


class CapucineError(Exception):
    """Racine de toutes les erreurs de Capucine."""


class ConfigError(CapucineError):
    """Fichier de configuration absent, illisible ou incohérent."""


class SchemaError(CapucineError):
    """Signature de skill impossible à traduire en schéma d'outil."""


class PluginError(CapucineError):
    """Problème imputable à un plugin, jamais au cœur."""


class PluginImportError(PluginError):
    """Le module a levé une exception à l'import."""

    def __init__(self, path: str, message: str, missing_package: str | None = None):
        super().__init__(message)
        self.path = path
        self.missing_package = missing_package


class SkillRefused(PluginError):
    """Le plugin refuse d'agir, pour une raison que l'utilisateur doit entendre.

    À distinguer d'un plantage : « ce fichier est hors de l'atelier » ou « il
    faut une clé d'API » ne sont pas des bogues, ce sont des réponses. Le
    registre les transmet telles quelles au lieu de les traduire en « je n'ai
    pas pu exécuter cette commande », qui n'apprend rien à personne.
    """


class SkillTimeout(PluginError):
    """Le skill a dépassé son délai d'exécution."""


class SkillCrashed(PluginError):
    """Le skill a levé une exception à l'exécution."""


class ArgumentError(PluginError):
    """Les arguments proposés ne correspondent pas au schéma du skill."""


class EngineUnavailable(CapucineError):
    """Un moteur (LLM, STT, TTS, wake word) n'est pas utilisable."""

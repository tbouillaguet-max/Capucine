"""Le contrat public offert aux plugins.

Un plugin est un fichier ``.py`` déposé dans ``plugins/``. Il n'a besoin que
de ce module :

    from capucine.plugin import skill

    @skill(description="…", examples=["…"])
    def ma_competence(argument: str) -> str:
        \"\"\"Cette docstring sert de contexte au LLM.\"\"\"
        return "réponse lue à voix haute"

Rien d'autre : pas d'enregistrement manuel, pas d'import à ajouter ailleurs,
pas de manifeste. Le module ``capucine.plugin`` n'est qu'une façade sur ce
fichier, pour que le cœur puisse bouger sans casser un seul plugin.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import SchemaError
from .logging import get_logger
from .schema import build_parameters_schema, build_tool_schema
from .text import ascii_identifier

__all__ = [
    "skill",
    "SkillDeclaration",
    "SkillSpec",
    "PluginContext",
    "get_config",
    "get_logger_for_plugin",
    "data_dir",
    "announce",
    "set_announcer",
    "register_context",
    "clear_context",
    "SKILL_ATTRIBUTE",
]

SKILL_ATTRIBUTE = "__capucine_skill__"


@dataclass(frozen=True)
class SkillDeclaration:
    """Ce que le décorateur attache à la fonction, avant tout chargement."""

    description: str
    examples: tuple[str, ...]
    name: str | None
    timeout: float | None
    confirm: bool


def skill(
    _func: Callable[..., Any] | None = None,
    *,
    description: str = "",
    examples: list[str] | tuple[str, ...] | None = None,
    name: str | None = None,
    timeout: float | None = None,
    confirm: bool = False,
) -> Callable[..., Any]:
    """Déclare une fonction comme compétence de Capucine.

    Args:
        description: Une phrase, en français, qui dit ce que fait la
            compétence. Elle part telle quelle dans le contexte du LLM.
        examples: Formulations typiques de l'utilisateur. Elles ne sont pas
            décoratives : le routeur déterministe s'en sert pour choisir
            l'outil sans solliciter le modèle.
        name: Nom d'outil, si le nom de la fonction ne convient pas.
        timeout: Délai maximum d'exécution, en secondes. Au-delà, Capucine
            répond qu'elle n'a pas pu exécuter la commande.
        confirm: Réservé aux actions irréversibles ; Capucine demandera
            confirmation avant d'exécuter (câblé à l'étape 4).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(func):  # pragma: no cover - garde-fou
            raise SchemaError("@skill s'applique à une fonction.")
        setattr(
            func,
            SKILL_ATTRIBUTE,
            SkillDeclaration(
                description=description.strip(),
                examples=tuple(examples or ()),
                name=name,
                timeout=timeout,
                confirm=confirm,
            ),
        )
        return func

    if _func is not None:
        return decorator(_func)
    return decorator


@dataclass
class SkillSpec:
    """Une compétence chargée et prête à être appelée."""

    name: str
    func: Callable[..., Any]
    module: str
    source: Path
    description: str
    examples: tuple[str, ...]
    timeout: float | None
    confirm: bool
    tool_schema: dict[str, Any]
    parameters_schema: dict[str, Any]
    plugin: str
    qualname: str = ""
    failures: int = 0
    quarantined: bool = False

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.func)

    @property
    def required_parameters(self) -> list[str]:
        return list(self.parameters_schema.get("required", []))

    @property
    def parameter_names(self) -> list[str]:
        return list(self.parameters_schema.get("properties", {}))

    def matchable_phrases(self) -> list[str]:
        """Ce que le routeur déterministe compare à la phrase entendue."""
        readable_name = self.name.replace("_", " ")
        return [readable_name, *self.examples, self.description]


def build_skill_spec(
    func: Callable[..., Any],
    declaration: SkillDeclaration,
    *,
    module: str,
    source: Path,
    plugin: str,
) -> SkillSpec:
    tool_name = ascii_identifier(declaration.name or func.__name__)
    parameters_schema, _ = build_parameters_schema(func)
    tool_schema = build_tool_schema(
        func,
        name=tool_name,
        description=declaration.description,
        examples=list(declaration.examples),
    )
    return SkillSpec(
        name=tool_name,
        func=func,
        module=module,
        source=source,
        description=tool_schema["function"]["description"],
        examples=declaration.examples,
        timeout=declaration.timeout,
        confirm=declaration.confirm,
        tool_schema=tool_schema,
        parameters_schema=parameters_schema,
        plugin=plugin,
        qualname=getattr(func, "__qualname__", func.__name__),
    )


# --- contexte d'un plugin --------------------------------------------------

@dataclass
class PluginContext:
    """Ce que le cœur met à disposition d'un plugin donné."""

    plugin: str
    module: str
    config: dict[str, Any] = field(default_factory=dict)
    data_dir: Path | None = None
    logger: logging.Logger | None = None

    def get_logger(self) -> logging.Logger:
        if self.logger is None:
            self.logger = get_logger(f"plugin.{self.plugin}")
        return self.logger


_CONTEXTS: dict[str, PluginContext] = {}
_ANNOUNCER: Callable[[str], None] | None = None
_logger = get_logger("plugin")


def register_context(context: PluginContext) -> None:
    _CONTEXTS[context.module] = context


def clear_context(module: str) -> None:
    _CONTEXTS.pop(module, None)


def _caller_context(depth: int = 2) -> PluginContext | None:
    """Retrouve le plugin appelant en remontant la pile.

    Fonctionne à l'import du module comme à l'exécution d'un skill, parce que
    le registre enregistre le contexte *avant* d'exécuter le fichier.
    """
    frame = inspect.currentframe()
    for _ in range(depth):
        if frame is None:
            return None
        frame = frame.f_back
    while frame is not None:
        module_name = frame.f_globals.get("__name__", "")
        context = _CONTEXTS.get(module_name)
        if context is not None:
            return context
        frame = frame.f_back
    return None


def get_config(key: str | None = None, default: Any = None) -> Any:
    """Configuration du plugin appelant.

    Sans argument, retourne le dictionnaire complet : ``CONFIG_DEFAULTS``
    du module surchargé par la section ``[plugins.<nom>]`` du fichier de
    configuration.
    """
    context = _caller_context()
    if context is None:
        _logger.debug("get_config() hors contexte de plugin.")
        return default if key is not None else {}
    if key is None:
        return dict(context.config)
    return context.config.get(key, default)


def get_logger_for_plugin() -> logging.Logger:
    """Journal nommé d'après le plugin appelant."""
    context = _caller_context()
    return context.get_logger() if context else get_logger("plugin.inconnu")


def data_dir() -> Path:
    """Dossier inscriptible réservé au plugin appelant, créé à la demande."""
    context = _caller_context()
    if context is None or context.data_dir is None:
        fallback = Path.home() / ".capucine" / "data" / "inconnu"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    context.data_dir.mkdir(parents=True, exist_ok=True)
    return context.data_dir


def set_announcer(announcer: Callable[[str], None] | None) -> None:
    """Branche la sortie vocale utilisée par ``announce()``.

    Le pipeline l'installe au démarrage ; en dehors, ``announce()`` se replie
    sur le journal, ce qui rend les plugins testables sans assistant.
    """
    global _ANNOUNCER
    _ANNOUNCER = announcer


def announce(message: str) -> None:
    """Fait dire quelque chose à Capucine hors d'un tour de conversation.

    C'est ce dont a besoin une tâche de fond : un minuteur qui arrive à
    échéance n'a personne à qui répondre, il doit *interrompre*.
    """
    if _ANNOUNCER is not None:
        try:
            _ANNOUNCER(message)
            return
        except Exception:  # pragma: no cover - une annonce ne casse jamais rien
            _logger.exception("Échec de l'annonce, repli sur le journal.")
    _logger.info("[annonce] %s", message)


def config_defaults(module: Any) -> Mapping[str, Any]:
    values = getattr(module, "CONFIG_DEFAULTS", {})
    return values if isinstance(values, Mapping) else {}

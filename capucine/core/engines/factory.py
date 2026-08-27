"""Fabrique de moteurs : passer d'un backend à l'autre est une ligne de config.

Les modules concrets sont importés **à l'instanciation seulement**. C'est ce
qui permet à ``python main.py --text`` de démarrer en une seconde sur un
Raspberry Pi sans charger torch, et aux tests de tourner sans la stack audio.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from ..errors import ConfigError, EngineUnavailable
from ..interfaces.llm import LLMEngine
from ..logging import get_logger

logger = get_logger("engines")

# nom de config -> (module, classe)
LLM_ENGINES: dict[str, tuple[str, str]] = {
    "mock": ("capucine.core.engines.llm.mock", "MockLLM"),
    "ollama": ("capucine.core.engines.llm.ollama", "OllamaLLM"),
    "llamacpp": ("capucine.core.engines.llm.llamacpp", "LlamaCppLLM"),
}

# Les étages audio arrivent aux étapes 2 et 3 ; les tables existent pour que
# leur branchement ne touche pas au reste du cœur.
STT_ENGINES: dict[str, tuple[str, str]] = {}
TTS_ENGINES: dict[str, tuple[str, str]] = {}
WAKE_ENGINES: dict[str, tuple[str, str]] = {}


def _instantiate(table: Mapping[str, tuple[str, str]], kind: str, name: str, options: Mapping[str, Any]) -> Any:
    if name not in table:
        known = ", ".join(sorted(table)) or "aucun pour l'instant"
        raise ConfigError(f"Moteur {kind} inconnu : « {name} ». Disponibles : {known}.")
    module_name, class_name = table[name]
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise EngineUnavailable(
            f"Le moteur {kind} « {name} » n'est pas installable en l'état : {exc}"
        ) from exc
    factory = getattr(module, class_name)
    return factory(**dict(options))


def build_llm(config: Any) -> LLMEngine:
    """Construit le moteur LLM décrit par ``[llm]``, avec repli sur le factice.

    Le repli est volontaire : une Capucine sans modèle doit continuer à
    exécuter des compétences plutôt que refuser de démarrer.
    """
    section = dict(config.section("llm"))
    name = str(section.pop("engine", "mock"))
    section.pop("router", None)
    try:
        engine = _instantiate(LLM_ENGINES, "LLM", name, section)
    except (ConfigError, EngineUnavailable) as exc:
        logger.error("%s", exc)
        logger.warning("Repli sur le moteur factice : les compétences resteront utilisables.")
        return _instantiate(LLM_ENGINES, "LLM", "mock", {})

    if not engine.available():
        logger.warning(
            "Moteur LLM « %s » injoignable. Repli sur le moteur factice : "
            "les compétences restent utilisables, la conversation libre non.",
            engine.describe(),
        )
        return _instantiate(LLM_ENGINES, "LLM", "mock", {})
    logger.info("Moteur LLM : %s", engine.describe())
    return engine

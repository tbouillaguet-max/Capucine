"""Assemblage : configuration → registre → routeur → pipeline.

C'est le seul endroit qui connaît tous les étages à la fois. Tout le reste du
cœur n'en voit qu'un.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .core.config import Config
from .core.conversation import Conversation, load_persona
from .core.engines.factory import build_llm
from .core.interfaces.llm import LLMEngine
from .core.logging import get_logger
from .core.pipeline import Pipeline
from .core.registry import PluginRegistry
from .core.router import Router

logger = get_logger("app")


@dataclass
class Assistant:
    config: Config
    llm: LLMEngine
    registry: PluginRegistry
    router: Router
    conversation: Conversation
    pipeline: Pipeline

    async def aclose(self) -> None:
        await self.pipeline.aclose()
        self.llm.close()


def build_assistant(config: Config, llm: LLMEngine | None = None) -> Assistant:
    engine = llm if llm is not None else build_llm(config)

    router_options = config.section("llm").get("router", {}) or {}
    router = Router(
        engine,
        direct_threshold=float(router_options.get("direct_threshold", 0.72)),
        shortlist_threshold=float(router_options.get("shortlist_threshold", 0.35)),
        shortlist_size=int(router_options.get("shortlist_size", 5)),
        allow_number_extraction=bool(router_options.get("number_extraction", True)),
    )

    conversation = Conversation(
        persona=load_persona(config.resolve_path("assistant.persona_file")),
        max_turns=int(config.get("assistant.memory_turns", 6)),
    )

    registry = PluginRegistry(
        config.plugin_paths(),
        config=config,
        default_timeout=float(config.get("plugins.timeout", 10.0)),
        data_root=config.resolve_path("plugins.data_dir"),
        quarantine_after=int(config.get("plugins.quarantine_after", 3)),
    )

    pipeline = Pipeline(
        registry,
        router,
        conversation,
        announce_new_skills=bool(config.get("assistant.announce_new_skills", True)),
    )
    registry.load_all()
    # Le rappel n'est branché qu'APRÈS le chargement initial : annoncer à voix
    # haute les vingt compétences déjà présentes au démarrage n'a aucun sens.
    # Le registre prévient le pipeline, qui décide quoi annoncer — le registre
    # n'a pas à savoir que Capucine a une voix.
    registry.on_change = pipeline.notify_skill_change
    return Assistant(
        config=config, llm=engine, registry=registry, router=router,
        conversation=conversation, pipeline=pipeline,
    )


# --- mode texte ------------------------------------------------------------

AIDE = """Commandes : /aide  /competences  /plugins  /recharge  /oublie  /quitter
Tout le reste est traité comme une phrase adressée à Capucine."""


def _format_skills(assistant: Assistant) -> str:
    skills = assistant.registry.skills
    if not skills:
        return "Aucune compétence chargée."
    lines = []
    for name, spec in sorted(skills.items()):
        signature = ", ".join(spec.parameter_names) or "sans argument"
        marque = " [en quarantaine]" if spec.quarantined else ""
        lines.append(f"  {name}({signature}) — {spec.plugin}{marque}")
    return "Compétences :\n" + "\n".join(lines)


def _format_plugins(assistant: Assistant) -> str:
    records = assistant.registry.plugins
    if not records:
        return "Aucun plugin trouvé. Vérifiez plugins.paths dans la configuration."
    lines = []
    for name, record in sorted(records.items()):
        if record.ok:
            lines.append(f"  ✓ {name} — {len(record.skills)} compétence(s) — {record.path}")
        else:
            lines.append(f"  ✗ {name} — {record.error}")
    return "Plugins :\n" + "\n".join(lines)


async def run_text_mode(assistant: Assistant, once: str | None = None) -> int:
    """Boucle clavier : court-circuite micro et haut-parleur.

    C'est le mode de développement : il prouve la boucle LLM + plugins sans
    qu'une seule ligne d'audio soit nécessaire.
    """
    assistant.pipeline.attach()
    try:
        if once is not None:
            result = await assistant.pipeline.handle_and_speak(once)
            return 0 if (result.skill_result is None or result.skill_result.ok) else 1

        print(f"Capucine — mode texte ({assistant.llm.describe()}, profil "
              f"{assistant.config.get('profile')}). /aide pour les commandes.")
        if assistant.llm.name == "mock":
            print("Moteur factice : le routage déterministe fonctionne, la conversation "
                  "libre non. Formulez les commandes au plus près des exemples des plugins.")
        print(_format_skills(assistant))
        for record in assistant.registry.failures():
            print(f"  ! plugin ignoré : {record.name} — {record.error}")

        while True:
            try:
                line = (await asyncio.to_thread(input, "\nVous  › ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line.startswith("/"):
                if _handle_command(assistant, line):
                    return 0
                continue
            await assistant.pipeline.handle_and_speak(line)
    finally:
        assistant.pipeline.detach()


def _handle_command(assistant: Assistant, line: str) -> bool:
    """Retourne True s'il faut quitter."""
    command = line.split()[0].lower()
    if command in ("/quitter", "/quit", "/q"):
        return True
    if command in ("/aide", "/help", "/h"):
        print(AIDE)
    elif command in ("/competences", "/skills"):
        print(_format_skills(assistant))
    elif command == "/plugins":
        print(_format_plugins(assistant))
    elif command in ("/recharge", "/reload"):
        assistant.registry.load_all()
        print(_format_skills(assistant))
    elif command in ("/oublie", "/clear"):
        assistant.conversation.clear()
        print("Mémoire de conversation vidée.")
    else:
        print(f"Commande inconnue : {command}\n{AIDE}")
    return False


def describe_startup(assistant: Assistant) -> dict[str, Any]:
    return {
        "profil": assistant.config.get("profile"),
        "llm": assistant.llm.describe(),
        "plugins": len([r for r in assistant.registry.plugins.values() if r.ok]),
        "competences": len(assistant.registry.skills),
        "echecs": len(assistant.registry.failures()),
    }

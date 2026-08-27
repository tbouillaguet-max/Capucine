"""La machine à états.

    IDLE → WAKE → LISTEN → TRANSCRIBE → THINK → ACT → SPEAK → IDLE

À l'étape 1, seul le chemin texte est câblé : ``THINK → ACT → SPEAK``. Les
états audio existent déjà et sont traversés par le pipeline vocal des étapes
2 et 3 sans que la structure change.

L'orchestration est en ``asyncio``, décision prise dès maintenant parce
qu'elle engage la suite : le barge-in, c'est l'annulation propre d'une chaîne
``THINK → SPEAK`` déjà lancée, et ``task.cancel()`` est autrement plus sûr que
des drapeaux ``threading.Event`` disséminés. Les plugins, eux, restent des
fonctions **synchrones** ordinaires exécutées dans un thread : le contrat de
plugin ne change pas d'un caractère.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .conversation import Conversation
from .interfaces.llm import ToolCall
from .logging import TurnTelemetry, get_logger
from .plugin import set_announcer
from .registry import PluginRegistry, SkillResult
from .router import RouteDecision, Router

logger = get_logger("pipeline")


class State(StrEnum):
    IDLE = "idle"
    WAKE = "wake"
    LISTEN = "listen"
    TRANSCRIBE = "transcribe"
    THINK = "think"
    ACT = "act"
    SPEAK = "speak"


@dataclass
class TurnResult:
    """Ce qu'un tour produit, quelle que soit son entrée (voix ou clavier)."""

    utterance: str
    speak: str = ""
    display: str = ""
    tier: str = "conversation"
    tool: ToolCall | None = None
    skill_result: SkillResult | None = None
    telemetry: TurnTelemetry = field(default_factory=TurnTelemetry)

    @property
    def text(self) -> str:
        return self.display or self.speak


class Pipeline:
    """Orchestre un tour, de la phrase entendue à la phrase prononcée."""

    def __init__(
        self,
        registry: PluginRegistry,
        router: Router,
        conversation: Conversation,
        *,
        speak: Callable[[str], Any] | None = None,
        on_state: Callable[[State], None] | None = None,
        announce_new_skills: bool = True,
    ) -> None:
        self.registry = registry
        self.router = router
        self.conversation = conversation
        self._speak = speak
        self._on_state = on_state
        self.announce_new_skills = announce_new_skills
        self._state = State.IDLE
        self._current: asyncio.Task[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._announcements: asyncio.Queue[str] = asyncio.Queue()

    # -- état ---------------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    def _set_state(self, state: State) -> None:
        self._state = state
        logger.debug("état → %s", state.value)
        if self._on_state:
            try:
                self._on_state(state)
            except Exception:  # pragma: no cover - un observateur ne casse rien
                logger.exception("Observateur d'état en échec.")

    # -- cycle de vie -------------------------------------------------------
    def attach(self) -> None:
        """Branche ``capucine.plugin.announce()`` sur la sortie de Capucine.

        Une tâche de fond — un minuteur qui sonne — n'a personne à qui
        répondre : elle doit pouvoir interrompre. Le rappel est thread-safe,
        car ces tâches tournent hors de la boucle asyncio.
        """
        self._loop = asyncio.get_running_loop()
        set_announcer(self._announce_threadsafe)

    def detach(self) -> None:
        set_announcer(None)
        self._loop = None

    def _announce_threadsafe(self, message: str) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            logger.info("[annonce] %s", message)
            return
        try:
            en_cours = asyncio.get_running_loop()
        except RuntimeError:
            en_cours = None
        if en_cours is loop:
            # Déjà sur la boucle (rechargement manuel, commande /recharge) :
            # `call_soon_threadsafe` ne serait traité qu'au prochain tour de
            # boucle, et une purge immédiate raterait l'annonce.
            self._announcements.put_nowait(message)
        else:
            loop.call_soon_threadsafe(self._announcements.put_nowait, message)

    async def drain_announcements(self) -> None:
        """Prononce les annonces en attente (minuteur, nouvelle compétence)."""
        while not self._announcements.empty():
            message = self._announcements.get_nowait()
            await self.say(message)

    def notify_skill_change(self, added: list[str], removed: list[str]) -> None:
        """Rappel branché sur le registre pour l'étape 4.

        Volontairement discret : on n'annonce que les compétences dont le *nom*
        est nouveau. Pendant le développement, on enregistre un fichier trente
        fois par heure ; une Capucine qui commente chaque sauvegarde devient
        vite insupportable.
        """
        for name in removed:
            logger.info("Compétence retirée : %s", name)
        if not self.announce_new_skills or not added:
            return
        for name in added:
            self._announce_threadsafe(f"Nouvelle compétence disponible : {name.replace('_', ' ')}.")

    # -- sortie -------------------------------------------------------------
    async def say(self, text: str) -> None:
        if not text:
            return
        self._set_state(State.SPEAK)
        try:
            if self._speak is None:
                print(f"Capucine › {text}")
            else:
                result = self._speak(text)
                if asyncio.iscoroutine(result):
                    await result
        finally:
            self._set_state(State.IDLE)

    # -- un tour ------------------------------------------------------------
    async def handle(self, utterance: str) -> TurnResult:
        """Traite une phrase et retourne le tour complet. Ne lève jamais."""
        telemetry = TurnTelemetry(name="tour")
        result = TurnResult(utterance=utterance, telemetry=telemetry)
        utterance = utterance.strip()
        if not utterance:
            return result

        self.conversation.add_user(utterance)
        skills = self.registry.skills

        self._set_state(State.THINK)
        try:
            with telemetry.stage("reflexion_ms"):
                decision: RouteDecision = await asyncio.to_thread(
                    self.router.route,
                    utterance,
                    skills,
                    self.conversation.history()[:-1],
                    self.conversation.system_prompt(),
                )
        except Exception as exc:
            logger.exception("Le routage a échoué.")
            result.speak = result.display = "Je n'ai pas réussi à traiter cette demande."
            result.tier = "erreur"
            self.conversation.add_assistant(result.speak)
            telemetry.emit(etage="erreur", erreur=type(exc).__name__)
            self._set_state(State.IDLE)
            return result

        result.tier = decision.tier
        result.tool = decision.tool_call

        if decision.tool_call is not None:
            self._set_state(State.ACT)
            with telemetry.stage("execution_ms"):
                skill_result = await asyncio.to_thread(
                    self.registry.call, decision.tool_call.name, decision.tool_call.arguments
                )
            result.skill_result = skill_result
            result.speak = skill_result.speak
            result.display = skill_result.display or skill_result.speak
            self.conversation.add_tool_result(skill_result.skill, result.display)
        else:
            with telemetry.stage("reponse_ms"):
                answer = await asyncio.to_thread(
                    self.router.answer_freely,
                    utterance,
                    self.conversation.history()[:-1],
                    self.conversation.system_prompt(),
                )
            result.speak = result.display = answer
            self.conversation.add_assistant(answer)

        telemetry.emit(
            etage=decision.tier,
            outil=decision.tool_call.name if decision.tool_call else "-",
            ok=result.skill_result.ok if result.skill_result else True,
        )
        self._set_state(State.IDLE)
        return result

    async def handle_and_speak(self, utterance: str) -> TurnResult:
        result = await self.handle(utterance)
        await self.say(result.speak)
        await self.drain_announcements()
        return result

    # -- barge-in (câblé à l'étape 3) ---------------------------------------
    def start_turn(self, utterance: str) -> asyncio.Task[TurnResult]:
        """Lance un tour annulable, et abandonne celui qui serait en cours."""
        self.cancel_turn()
        task = asyncio.ensure_future(self.handle_and_speak(utterance))
        self._current = task
        return task

    def cancel_turn(self) -> bool:
        """Interrompt le tour en cours. C'est la mécanique du barge-in."""
        task = self._current
        if task is not None and not task.done():
            task.cancel()
            self._current = None
            self._set_state(State.IDLE)
            return True
        self._current = None
        return False

    async def aclose(self) -> None:
        self.cancel_turn()
        if self._current is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._current
        self.detach()

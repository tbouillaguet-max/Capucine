"""La machine à états.

    IDLE → WAKE → LISTEN → TRANSCRIBE → THINK → ACT → SPEAK → IDLE

À l'étape 2, tout le chemin est câblé sauf ``WAKE`` : la capture est déclenchée
au clavier, l'étape 3 la déclenchera au mot d'éveil et terminera l'énoncé au
VAD. Rien d'autre ne bougera.

L'orchestration est en ``asyncio``, décision prise dès l'étape 1 parce qu'elle
engage la suite : le barge-in, c'est l'annulation propre d'une chaîne
``THINK → SPEAK`` déjà lancée. Les plugins, eux, restent des fonctions
**synchrones** ordinaires exécutées dans un thread : le contrat de plugin ne
change pas d'un caractère.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .audio import AudioBuffer, AudioInput, AudioOutput, AudioUnavailable, record
from .conversation import Conversation
from .interfaces.llm import ToolCall
from .interfaces.stt import STTEngine, Transcription
from .interfaces.tts import TTSEngine
from .logging import TurnTelemetry, get_logger
from .plugin import set_announcer
from .registry import PluginRegistry, SkillResult
from .router import RouteDecision, Router
from .text import stream_sentences

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
    interrupted: bool = False

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
        stt: STTEngine | None = None,
        tts: TTSEngine | None = None,
        audio_in: AudioInput | None = None,
        audio_out: AudioOutput | None = None,
        speak: Callable[[str], Any] | None = None,
        on_state: Callable[[State], None] | None = None,
        announce_new_skills: bool = True,
        max_utterance_s: float = 20.0,
        echo: bool = True,
    ) -> None:
        self.registry = registry
        self.router = router
        self.conversation = conversation
        self.stt = stt
        self.tts = tts
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.announce_new_skills = announce_new_skills
        self.max_utterance_s = max_utterance_s
        self.echo = echo

        self._speak = speak
        self._on_state = on_state
        self._state = State.IDLE
        self._current: asyncio.Task[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._announcements: asyncio.Queue[str] = asyncio.Queue()
        # Armé pour couper la parole en cours ; consulté par la synthèse entre
        # deux phrases et par la lecture entre deux tranches.
        self._barge_in = threading.Event()

    # -- état ---------------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    @property
    def has_voice(self) -> bool:
        return self.tts is not None and self.audio_out is not None

    def _set_state(self, state: State) -> None:
        if state is self._state:
            return  # un observateur n'a que faire des transitions immobiles
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
            await self.say(self._announcements.get_nowait())

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

    # -- écoute -------------------------------------------------------------
    async def listen(self, stop: threading.Event | None = None) -> AudioBuffer:
        """Capte un énoncé. À l'étape 3, ``stop`` sera armé par le VAD."""
        if self.audio_in is None:
            return AudioBuffer(b"", 16000)
        self._set_state(State.LISTEN)
        try:
            return await asyncio.to_thread(
                record, self.audio_in, stop=stop, max_seconds=self.max_utterance_s
            )
        finally:
            self._set_state(State.IDLE)

    async def transcribe(self, audio: AudioBuffer, telemetry: TurnTelemetry) -> Transcription:
        if self.stt is None or not audio:
            return Transcription("")
        self._set_state(State.TRANSCRIBE)
        try:
            with telemetry.stage("transcription_ms"):
                return await asyncio.to_thread(self.stt.transcribe, audio)
        finally:
            self._set_state(State.IDLE)

    # -- parole -------------------------------------------------------------
    async def say(self, text: str) -> bool:
        """Prononce un texte. Retourne ``False`` s'il a été interrompu."""
        if not text:
            return True
        return await self.say_stream([text])

    async def say_stream(self, fragments: Iterable[str]) -> bool:
        """Prononce un flux de fragments, phrase par phrase.

        C'est là que se gagne la latence perçue : dès qu'une phrase est
        complète, elle part à la synthèse puis au haut-parleur, pendant que le
        modèle écrit encore la suivante.
        """
        self._barge_in.clear()
        self._set_state(State.SPEAK)
        try:
            if not self.has_voice:
                # Sans voix, on attend le texte complet : le découper n'aurait
                # aucun intérêt puisqu'il s'affiche d'un bloc.
                phrases = await asyncio.to_thread(lambda: list(stream_sentences(fragments)))
                texte = " ".join(phrases).strip()
                if texte:
                    await self._dire_en_texte(texte)
                return True

            interrompu = False
            async for phrase in _phrases_async(fragments):
                if self._barge_in.is_set():
                    interrompu = True
                    break
                if self.echo:
                    print(f"Capucine › {phrase}")
                if not await asyncio.to_thread(self._synthetiser_et_jouer, phrase):
                    interrompu = True
                    break
            return not interrompu
        finally:
            self._set_state(State.IDLE)

    def _synthetiser_et_jouer(self, phrase: str) -> bool:
        """Synthèse puis lecture d'une phrase, dans un thread. Ne lève jamais."""
        assert self.tts is not None and self.audio_out is not None
        try:
            for chunk in self.tts.synthesize(phrase, cancel=self._barge_in):
                if self._barge_in.is_set():
                    return False
                if not self.audio_out.play(chunk, cancel=self._barge_in):
                    return False
        except AudioUnavailable as exc:
            # Le périphérique a disparu en cours de route : on bascule en
            # affichage pour le reste de la session plutôt que de répéter
            # l'erreur à chaque phrase.
            logger.warning("Sortie audio perdue (%s) ; Capucine affichera ses réponses.", exc)
            self.audio_out = None
            print(f"Capucine › {phrase}")
            return True
        except Exception:
            logger.exception("Synthèse ou lecture en échec ; la réponse reste affichée.")
            print(f"Capucine › {phrase}")
            return True
        return True

    async def _dire_en_texte(self, texte: str) -> None:
        if self._speak is not None:
            resultat = self._speak(texte)
            if asyncio.iscoroutine(resultat):
                await resultat
        else:
            print(f"Capucine › {texte}")

    # -- un tour ------------------------------------------------------------
    async def _router(self, utterance: str, telemetry: TurnTelemetry) -> RouteDecision | None:
        self._set_state(State.THINK)
        try:
            with telemetry.stage("reflexion_ms"):
                return await asyncio.to_thread(
                    self.router.route,
                    utterance,
                    self.registry.skills,
                    self.conversation.history()[:-1],
                    self.conversation.system_prompt(),
                )
        except Exception:
            logger.exception("Le routage a échoué.")
            return None

    async def _executer(
        self, decision: RouteDecision, result: TurnResult, telemetry: TurnTelemetry
    ) -> None:
        assert decision.tool_call is not None
        self._set_state(State.ACT)
        with telemetry.stage("execution_ms"):
            skill_result = await asyncio.to_thread(
                self.registry.call, decision.tool_call.name, decision.tool_call.arguments
            )
        result.skill_result = skill_result
        result.speak = skill_result.speak
        result.display = skill_result.display or skill_result.speak
        self.conversation.add_tool_result(skill_result.skill, result.display)

    def _echec(self, result: TurnResult, telemetry: TurnTelemetry) -> TurnResult:
        result.speak = result.display = "Je n'ai pas réussi à traiter cette demande."
        result.tier = "erreur"
        self.conversation.add_assistant(result.speak)
        telemetry.emit(etage="erreur")
        self._set_state(State.IDLE)
        return result

    async def handle(self, utterance: str) -> TurnResult:
        """Traite une phrase sans la prononcer. Ne lève jamais."""
        telemetry = TurnTelemetry(name="tour")
        result = TurnResult(utterance=utterance.strip(), telemetry=telemetry)
        if not result.utterance:
            return result

        self.conversation.add_user(result.utterance)
        decision = await self._router(result.utterance, telemetry)
        if decision is None:
            return self._echec(result, telemetry)

        result.tier = decision.tier
        result.tool = decision.tool_call

        if decision.tool_call is not None:
            await self._executer(decision, result, telemetry)
        else:
            try:
                with telemetry.stage("reponse_ms"):
                    reponse = await asyncio.to_thread(
                        self.router.answer_freely,
                        result.utterance,
                        self.conversation.history()[:-1],
                        self.conversation.system_prompt(),
                    )
            except Exception:
                logger.exception("La génération de la réponse a échoué.")
                return self._echec(result, telemetry)
            result.speak = result.display = reponse
            self.conversation.add_assistant(reponse)

        telemetry.emit(
            etage=decision.tier,
            outil=decision.tool_call.name if decision.tool_call else "-",
            ok=result.skill_result.ok if result.skill_result else True,
        )
        self._set_state(State.IDLE)
        return result

    async def handle_and_speak(self, utterance: str) -> TurnResult:
        """Traite une phrase et la prononce.

        Sur le chemin conversationnel avec une voix disponible, la réponse est
        **diffusée** : la première phrase est prononcée pendant que le modèle
        écrit la suivante.
        """
        telemetry = TurnTelemetry(name="tour")
        result = TurnResult(utterance=utterance.strip(), telemetry=telemetry)
        if not result.utterance:
            return result

        self.conversation.add_user(result.utterance)
        decision = await self._router(result.utterance, telemetry)
        if decision is None:
            result = self._echec(result, telemetry)
            await self.say(result.speak)
            return result

        result.tier = decision.tier
        result.tool = decision.tool_call

        if decision.tool_call is not None:
            await self._executer(decision, result, telemetry)
            result.interrupted = not await self.say(result.speak)
        elif self.has_voice:
            morceaux: list[str] = []

            def fragments() -> Iterator[str]:
                for fragment in self.router.stream_answer(
                    result.utterance,
                    self.conversation.history()[:-1],
                    self.conversation.system_prompt(),
                ):
                    morceaux.append(fragment)
                    yield fragment

            with telemetry.stage("reponse_ms"):
                result.interrupted = not await self.say_stream(fragments())
            result.speak = result.display = "".join(morceaux).strip()
            self.conversation.add_assistant(result.display)
        else:
            with telemetry.stage("reponse_ms"):
                try:
                    reponse = await asyncio.to_thread(
                        self.router.answer_freely,
                        result.utterance,
                        self.conversation.history()[:-1],
                        self.conversation.system_prompt(),
                    )
                except Exception:
                    logger.exception("La génération de la réponse a échoué.")
                    result = self._echec(result, telemetry)
                    await self.say(result.speak)
                    return result
            result.speak = result.display = reponse
            self.conversation.add_assistant(reponse)
            await self.say(reponse)

        telemetry.emit(
            etage=decision.tier,
            outil=decision.tool_call.name if decision.tool_call else "-",
            ok=result.skill_result.ok if result.skill_result else True,
            interrompu=result.interrupted,
        )
        self._set_state(State.IDLE)
        await self.drain_announcements()
        return result

    async def voice_turn(self, stop: threading.Event | None = None) -> TurnResult:
        """Un tour vocal complet : écoute, transcription, réflexion, parole."""
        audio = await self.listen(stop=stop)
        telemetry = TurnTelemetry(name="tour vocal")
        transcription = await self.transcribe(audio, telemetry)
        if not transcription:
            logger.info("Rien à transcrire (%.2f s captées).", audio.duration_s)
            return TurnResult(utterance="", telemetry=telemetry)

        print(f"Vous  › {transcription.text}")
        result = await self.handle_and_speak(transcription.text)
        # Les latences d'écoute et de transcription appartiennent au même tour.
        for etage, valeur in telemetry.stages.items():
            result.telemetry.record(etage, valeur)
        result.telemetry.record("audio_s", round(audio.duration_s * 1000, 1))
        return result

    # -- barge-in -----------------------------------------------------------
    def start_turn(self, utterance: str) -> asyncio.Task[TurnResult]:
        """Lance un tour annulable, et abandonne celui qui serait en cours."""
        self.cancel_turn()
        task = asyncio.ensure_future(self.handle_and_speak(utterance))
        self._current = task
        return task

    def cancel_turn(self) -> bool:
        """Interrompt le tour en cours. C'est la mécanique du barge-in.

        Deux mouvements : on arme le drapeau, ce qui coupe la synthèse et la
        lecture entre deux tranches, et on annule la tâche asyncio, ce qui
        déroule proprement le reste du tour.
        """
        self._barge_in.set()
        if self.audio_out is not None:
            self.audio_out.stop()
        task = self._current
        if task is not None and not task.done():
            task.cancel()
            self._current = None
            self._set_state(State.IDLE)
            return True
        self._current = None
        return False

    async def warmup(self) -> None:
        """Charge les modèles avant le premier tour."""
        for moteur in (self.stt, self.tts):
            if moteur is not None:
                await asyncio.to_thread(moteur.warmup)

    async def aclose(self) -> None:
        self.cancel_turn()
        if self._current is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._current
        for ressource in (self.audio_in, self.audio_out, self.stt, self.tts):
            if ressource is not None:
                with contextlib.suppress(Exception):
                    ressource.close()
        self.detach()


async def _phrases_async(fragments: Iterable[str]) -> AsyncIterator[str]:
    """Rend les phrases une à une, sans bloquer la boucle.

    Le flux sous-jacent est bloquant — il tire des jetons du modèle — donc
    chaque avancée du générateur part dans un thread. C'est ce qui permet
    d'être en train de parler pendant que le modèle écrit la suite.
    """
    generateur = stream_sentences(fragments)
    sentinelle = object()

    def suivante() -> Any:
        return next(generateur, sentinelle)

    while True:
        phrase = await asyncio.to_thread(suivante)
        if phrase is sentinelle:
            return
        yield phrase

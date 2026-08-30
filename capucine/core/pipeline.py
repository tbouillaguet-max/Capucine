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

from .audio import (
    AudioBuffer,
    AudioChunk,
    AudioInput,
    AudioOutput,
    AudioUnavailable,
    record,
)
from .conversation import Conversation
from .endpointer import Utterance
from .interfaces.llm import ToolCall
from .interfaces.stt import STTEngine, Transcription
from .interfaces.tts import TTSEngine
from .listener import ListenerEvent, ListenMode, VoiceListener
from .logging import TurnTelemetry, get_logger
from .plugin import set_announcer
from .registry import PluginRegistry, SkillResult
from .router import RouteDecision, Router
from .text import accord_ou_refus, est_une_correction, stream_sentences

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
        follow_up_s: float = 8.0,
        wake_beep: bool = True,
        apprentissage: Any = None,
        connaissances: Any = None,
        corpus: Any = None,
        journal: Any = None,
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
        self.follow_up_s = follow_up_s
        self.wake_beep = wake_beep
        self.apprentissage = apprentissage
        self.connaissances = connaissances
        # Le corpus d'éveil attend un verdict après chaque déclenchement :
        # ce qui suit dit si elle a eu raison de se réveiller.
        self.corpus = corpus
        self._eveil_a_etiqueter = False
        # Les derniers gestes réussis : de quoi apprendre une routine en
        # disant « retiens ça » après les avoir faits.
        self.journal = journal
        # Dernier tour ayant appelé un outil : ce que corrigera un éventuel
        # « non, je voulais dire… ».
        self._dernier_routage: tuple[str, str] | None = None
        # L'amorce d'origine, à laquelle s'ajoute le vocabulaire appris.
        self._amorce_stt = getattr(stt, "initial_prompt", "") or ""

        self._speak = speak
        self._on_state = on_state
        self._state = State.IDLE
        self._current: asyncio.Task[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._announcements: asyncio.Queue[str] = asyncio.Queue()
        # Armé pour couper la parole en cours ; consulté par la synthèse entre
        # deux phrases et par la lecture entre deux tranches.
        self._barge_in = threading.Event()
        self._listener: VoiceListener | None = None
        # Compétence irréversible en attente d'un « oui » : (nom, arguments).
        self._confirmation: tuple[str, dict[str, Any]] | None = None
        self._events: asyncio.Queue[ListenerEvent] = asyncio.Queue()

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

    async def run_announcer(self, stop: asyncio.Event | None = None) -> None:
        """Prononce les annonces dès qu'elles arrivent, sans attendre un tour.

        Sans cette tâche de fond, un minuteur qui sonne pendant que Capucine
        est au repos resterait muet jusqu'à la prochaine phrase de
        l'utilisateur — ce qui vide un minuteur de son intérêt.

        On patiente tant qu'elle parle ou qu'elle exécute une compétence :
        interrompre l'utilisateur est le but, se couper soi-même ne l'est pas.
        """
        occupee = (State.THINK, State.ACT, State.SPEAK)
        while stop is None or not stop.is_set():
            message = await self._announcements.get()
            for _ in range(600):   # une minute d'attente au maximum
                if self._state not in occupee:
                    break
                await asyncio.sleep(0.1)
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

    # -- écoute -------------------------------------------------------------
    async def listen(self, stop: threading.Event | None = None) -> AudioBuffer:
        """Capte un énoncé. À l'étape 3, ``stop`` sera armé par le VAD."""
        if self.audio_in is None:
            return AudioBuffer(b"", 16000)
        self._set_state(State.LISTEN)
        # Pas de retour à IDLE ici : l'étage suivant enchaîne. Un observateur
        # n'a que faire d'un passage au repos qui dure une microseconde.
        return await asyncio.to_thread(
            record, self.audio_in, stop=stop, max_seconds=self.max_utterance_s
        )

    async def transcribe(self, audio: AudioBuffer, telemetry: TurnTelemetry) -> Transcription:
        if self.stt is None or not audio:
            return Transcription("")
        self._set_state(State.TRANSCRIBE)
        self._souffler_le_vocabulaire()
        with telemetry.stage("transcription_ms"):
            return await asyncio.to_thread(self.stt.transcribe, audio)

    def _souffler_le_vocabulaire(self) -> None:
        """Ajoute le vocabulaire appris à l'amorce de Whisper.

        Sans cela « CalculRisque » devient « calcul risque » à chaque fois — et
        la commande qui en dépend rate à chaque fois.
        """
        if self.apprentissage is None or self.stt is None:
            return
        if not hasattr(self.stt, "initial_prompt"):
            return
        try:
            self.stt.initial_prompt = self.apprentissage.amorce_stt(self._amorce_stt)
        except Exception:  # pragma: no cover
            logger.debug("Amorce de transcription inchangée.", exc_info=True)

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
        self._surveiller_interruption(True)
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
            self._surveiller_interruption(False)
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

    def _surveiller_interruption(self, actif: bool) -> None:
        """Ouvre (ou referme) la surveillance du micro pendant la parole.

        Le micro reste ouvert en permanence : c'est ce qui rend le barge-in
        possible. À la fin de la phrase on ne repasse en pause que si la
        surveillance est encore active — si l'utilisateur a déjà coupé la
        parole, la boucle a rouvert l'écoute et il ne faut pas la refermer.
        """
        listener = self._listener
        if listener is None or self.audio_out is None:
            return
        if actif:
            listener.set_mode(ListenMode.MONITOR)
        elif listener.mode is ListenMode.MONITOR and listener.pending is None:
            listener.set_mode(ListenMode.PAUSED)

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
        self._memoriser_confirmation(result)
        if self.journal is not None and skill_result.ok and not skill_result.needs_confirmation:
            self.journal.noter(decision.tool_call.name, decision.tool_call.arguments)

    async def _traiter_confirmation(
        self, result: TurnResult, telemetry: TurnTelemetry
    ) -> bool:
        """Interprète la phrase comme une réponse à une question en attente.

        Une compétence déclarée ``confirm=`` n'est pas exécutée du premier
        coup : Capucine pose la question, et le tour suivant est lu comme un
        oui ou un non. Toute autre réponse annule l'attente et repart en
        routage normal — on ne piège pas l'utilisateur dans une question.

        Retourne ``True`` si le tour a été entièrement traité ici.
        """
        if self._confirmation is None:
            return False
        nom, arguments = self._confirmation
        reponse = accord_ou_refus(result.utterance)

        if reponse is None:
            logger.debug("Confirmation abandonnée : la réponse ne tranche pas.")
            self._confirmation = None
            return False

        self._confirmation = None
        result.tier = "confirmation"
        if not reponse:
            result.speak = result.display = "Très bien, je n'ai rien fait."
            self.conversation.add_assistant(result.speak)
            telemetry.emit(etage="confirmation", outil=nom, ok=True, confirme=False)
            self._set_state(State.IDLE)
            return True

        self._set_state(State.ACT)
        with telemetry.stage("execution_ms"):
            skill_result = await asyncio.to_thread(
                self.registry.call, nom, arguments, confirmed=True
            )
        result.skill_result = skill_result
        result.tool = ToolCall(name=nom, arguments=arguments, source="confirmation")
        result.speak = skill_result.speak
        result.display = skill_result.display or skill_result.speak
        self.conversation.add_tool_result(skill_result.skill, result.display)
        telemetry.emit(etage="confirmation", outil=nom, ok=skill_result.ok, confirme=True)
        self._set_state(State.IDLE)
        return True

    def _memoriser_confirmation(self, result: TurnResult) -> bool:
        """Retient la compétence en attente si elle réclame un accord."""
        skill_result = result.skill_result
        if skill_result is None or not skill_result.needs_confirmation:
            return False
        self._confirmation = (skill_result.skill, dict(skill_result.arguments))
        logger.info("En attente d'une confirmation pour « %s ».", skill_result.skill)
        return True

    # -- apprentissage ------------------------------------------------------
    def _preparer_correction(self, utterance: str) -> tuple[str, str] | None:
        """Le tour précédent est-il en train d'être corrigé ?"""
        if self.apprentissage is None or not self.apprentissage.corrections_actives:
            return None
        if self._dernier_routage is None or not est_une_correction(utterance):
            return None
        return self._dernier_routage

    def _indexer_le_tour(self, result: TurnResult) -> None:
        """Confie le tour à l'index sémantique, qui le vectorise en fond.

        Hors du chemin chaud, et sans jamais lever : un index en panne ne doit
        pas coûter une réponse.
        """
        if self.connaissances is None or not result.display:
            return
        try:
            session = getattr(self.conversation, "session_id", None)
            self.connaissances.indexer_le_tour(
                f"conversation {session}" if session else "conversation",
                result.utterance, result.display,
            )
        except Exception:  # pragma: no cover - indexer ne casse jamais un tour
            logger.exception("Mise en file du tour pour indexation impossible.")

    def _apprendre_du_tour(
        self,
        result: TurnResult,
        decision: RouteDecision,
        a_corriger: tuple[str, str] | None,
    ) -> None:
        """Retient ce que ce tour a appris. Ne fait jamais échouer un tour."""
        self._indexer_le_tour(result)
        if self.apprentissage is None:
            return
        try:
            self.apprentissage.moissonner(result.utterance)
            if result.display:
                self.apprentissage.moissonner(result.display, source="reponse")

            outil = decision.tool_call.name if decision.tool_call else None
            reussi = outil is not None and (
                result.skill_result is None or result.skill_result.ok
            )
            if not reussi:
                return

            if a_corriger is not None:
                # « Non, je voulais dire le minuteur » : on désapprend ce qui
                # était faux ET on apprend ce qui était juste, sur la phrase
                # d'origine — c'est elle qui reviendra, pas la correction.
                phrase_initiale, ancien_outil = a_corriger
                if ancien_outil != outil:
                    self.apprentissage.dementir_routage(phrase_initiale, ancien_outil)
                    self.apprentissage.apprendre_routage(phrase_initiale, outil)
                    logger.info(
                        "Correction retenue : « %s » %s → %s",
                        phrase_initiale[:50], ancien_outil, outil,
                    )
            elif decision.tier == "llm":
                # L'étage déterministe a raté, le modèle a tranché : la
                # prochaine fois, l'étage déterministe saura.
                self.apprentissage.apprendre_routage(result.utterance, outil)

            self._dernier_routage = (result.utterance, outil)
        except Exception:  # pragma: no cover - apprendre ne casse jamais un tour
            logger.exception("Apprentissage du tour impossible.")

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
        if await self._traiter_confirmation(result, telemetry):
            return result

        a_corriger = self._preparer_correction(result.utterance)
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

        self._apprendre_du_tour(result, decision, a_corriger)
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
        if await self._traiter_confirmation(result, telemetry):
            await self.say(result.speak)
            return result

        a_corriger = self._preparer_correction(result.utterance)
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

        self._apprendre_du_tour(result, decision, a_corriger)
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
            self._set_state(State.IDLE)
            return TurnResult(utterance="", telemetry=telemetry)

        print(f"Vous  › {transcription.text}")
        result = await self.handle_and_speak(transcription.text)
        # Les latences d'écoute et de transcription appartiennent au même tour.
        for etage, valeur in telemetry.stages.items():
            result.telemetry.record(etage, valeur)
        result.telemetry.record("audio_s", round(audio.duration_s * 1000, 1))
        return result

    # -- boucle « toujours à l'écoute » -------------------------------------
    def on_listener_event(self, evenement: ListenerEvent) -> None:
        """Passe un événement du thread d'écoute à la boucle asyncio.

        Une exception, et elle est essentielle : le **barge-in coupe la parole
        immédiatement, depuis le thread d'écoute**, avant même d'être mis en
        file. La boucle de conversation, elle, est en train d'attendre la fin
        de ``handle_and_speak`` — elle ne dépilerait l'événement qu'une fois la
        réponse entièrement prononcée, c'est-à-dire trop tard pour
        l'interrompre. Le drapeau et l'arrêt du haut-parleur sont des objets de
        thread : on les actionne ici, tout de suite. La file ne sert plus qu'à
        décider de la suite — rouvrir l'écoute.
        """
        if evenement.kind == "barge_in":
            self._barge_in.set()
            if self.audio_out is not None:
                try:
                    self.audio_out.stop()
                except Exception:  # pragma: no cover - un arrêt raté ne casse rien
                    logger.debug("Arrêt de la sortie audio en erreur.", exc_info=True)

        loop = self._loop
        if loop is None or loop.is_closed():
            logger.debug("Événement d'écoute ignoré (%s) : pas de boucle.", evenement.kind)
            return
        loop.call_soon_threadsafe(self._events.put_nowait, evenement)

    async def run_conversation(
        self,
        listener: VoiceListener,
        *,
        use_wake: bool = True,
        stop: asyncio.Event | None = None,
    ) -> None:
        """Boucle complète : éveil, énoncé, réponse, suivi.

        C'est ici que la machine à états devient circulaire ::

            IDLE ──« Capucine »──▶ WAKE ─▶ LISTEN ─▶ TRANSCRIBE ─▶ THINK
                                                                     │
            IDLE ◀── suivi expiré ──── SPEAK ◀───── ACT ◀────────────┘
              ▲                          │
              └──── barge-in ────▶ LISTEN

        Le mode suivi garde l'écoute ouverte quelques secondes après la
        réponse : on enchaîne sans redire « Capucine ».
        """
        self._listener = listener
        mode_repos = ListenMode.WAKE if use_wake else ListenMode.UTTERANCE
        listener.set_mode(mode_repos)
        self._set_state(State.IDLE)

        while stop is None or not stop.is_set():
            evenement = await self._events.get()

            if evenement.kind == "stopped":
                logger.info("L'écoute s'est arrêtée.")
                return

            if evenement.kind == "wake" and evenement.wake is not None:
                self._set_state(State.WAKE)
                logger.info("Éveil : %s (%.2f)", evenement.wake.word, evenement.wake.score)
                self._eveil_a_etiqueter = True
                await self._bip()
                listener.endpointer.max_wait_s = self.max_utterance_s
                listener.set_mode(ListenMode.UTTERANCE)
                self._set_state(State.LISTEN)
                continue

            if evenement.kind == "barge_in":
                logger.info("On me coupe la parole : j'écoute.")
                self.cancel_turn()
                listener.endpointer.max_wait_s = self.max_utterance_s
                listener.set_mode(ListenMode.UTTERANCE)
                self._set_state(State.LISTEN)
                continue

            if evenement.kind != "utterance" or evenement.utterance is None:
                continue

            enonce = evenement.utterance
            if not enonce:
                # Fini, mais rien d'exploitable : silence, ou bruit trop bref.
                logger.debug("Énoncé sans contenu (%s).", enonce.reason.value)
                # Réveillée pour rien : c'est l'exemple négatif le plus
                # précieux qui soit, et il ne s'invente pas en studio.
                self._etiqueter_l_eveil(bon=False)
                listener.set_mode(mode_repos)
                self._set_state(State.IDLE)
                continue

            await self._tour_depuis_enonce(listener, enonce, mode_repos)

    async def _tour_depuis_enonce(
        self, listener: VoiceListener, enonce: Utterance, mode_repos: ListenMode
    ) -> None:
        """Transcrit, répond, puis ouvre la fenêtre de suivi."""
        listener.set_mode(ListenMode.PAUSED)
        telemetrie = TurnTelemetry(name="tour vocal")
        transcription = await self.transcribe(enonce.audio, telemetrie)

        if not transcription:
            logger.info("Rien de transcrit (%.2f s captées).", enonce.audio.duration_s)
            self._etiqueter_l_eveil(bon=False)
            listener.set_mode(mode_repos)
            self._set_state(State.IDLE)
            return

        self._etiqueter_l_eveil(bon=True)
        print(f"Vous  › {transcription.text}")
        resultat = await self.handle_and_speak(transcription.text)
        for etage, valeur in telemetrie.stages.items():
            resultat.telemetry.record(etage, valeur)
        resultat.telemetry.record("audio_s", round(enonce.audio.duration_s * 1000, 1))

        if listener.mode is ListenMode.UTTERANCE or listener.pending is ListenMode.UTTERANCE:
            # Le barge-in a déjà rouvert l'écoute : on ne la referme pas.
            return

        if self.follow_up_s > 0:
            # Mode suivi : quelques secondes pendant lesquelles on enchaîne
            # sans redire « Capucine ».
            listener.endpointer.max_wait_s = self.follow_up_s
            listener.set_mode(ListenMode.UTTERANCE)
            self._set_state(State.LISTEN)
        else:
            listener.set_mode(mode_repos)
            self._set_state(State.IDLE)

    def _etiqueter_l_eveil(self, *, bon: bool) -> None:
        """Dit au corpus si le dernier déclenchement était justifié.

        Sans rien demander à l'utilisateur : un énoncé transcrit derrière un
        éveil vaut confirmation, un silence vaut démenti. Ne lève jamais — un
        corpus en panne ne coûte pas un tour.
        """
        if not self._eveil_a_etiqueter:
            return
        self._eveil_a_etiqueter = False
        if self.corpus is None:
            return
        try:
            self.corpus.confirmer() if bon else self.corpus.dementir()
        except Exception:  # pragma: no cover - le corpus ne casse jamais un tour
            logger.exception("Étiquetage du corpus d'éveil impossible.")

    async def _bip(self) -> None:
        """Deux notes brèves pour dire « je t'écoute ».

        Plus rapide et moins bavard qu'un « oui ? » synthétisé : le signal
        doit arriver avant que l'utilisateur commence sa phrase, pas après.
        """
        if not self.wake_beep or self.audio_out is None:
            return
        try:
            await asyncio.to_thread(self.audio_out.play, carillon())
        except Exception:  # pragma: no cover - un bip raté n'empêche rien
            logger.debug("Bip d'éveil impossible.", exc_info=True)

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


def carillon(sample_rate: int = 16000) -> AudioChunk:
    """Le petit signal d'éveil, synthétisé à la volée — pas de fichier à livrer."""
    import math
    import struct

    echantillons: list[int] = []
    for frequence, duree in ((880.0, 0.07), (1320.0, 0.09)):
        n = int(sample_rate * duree)
        for i in range(n):
            # Enveloppe en cloche : sans elle, les bords claquent.
            enveloppe = math.sin(math.pi * i / n) ** 2
            echantillons.append(
                int(6000 * enveloppe * math.sin(2 * math.pi * frequence * i / sample_rate))
            )
    return AudioChunk(
        pcm=struct.pack(f"<{len(echantillons)}h", *echantillons),
        sample_rate=sample_rate,
    )


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


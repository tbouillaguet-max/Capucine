"""Le fil qui tient le micro.

Un seul thread ouvre l'entrée audio et la garde ouverte du début à la fin de
la session. C'est ce qui rend le barge-in possible : le micro n'est jamais
fermé, même pendant que Lily parle. Selon le mode courant, chaque trame
part vers le détecteur de mot d'éveil, vers le découpeur d'énoncé, ou vers la
surveillance d'interruption.

Le thread ne décide de rien : il **émet des événements** que la boucle asyncio
consomme. Les changements de mode viennent en sens inverse. Aucun verrou : le
mode demandé est appliqué en tête de boucle, entre deux trames, jamais au
milieu du traitement de l'une d'elles.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .audio import AudioInput, Rechunker
from .endpointer import BargeInDetector, Endpointer, Utterance
from .interfaces.wake import WakeEvent, WakeWordEngine
from .logging import get_logger

logger = get_logger("ecoute")


class ListenMode(StrEnum):
    PAUSED = "pause"            # les trames sont jetées (réflexion, exécution)
    WAKE = "eveil"              # on attend « Lily »
    UTTERANCE = "enonce"        # on capture ce qui est dit
    MONITOR = "surveillance"    # Lily parle : on guette une interruption


class BargeInMode(StrEnum):
    OFF = "off"                 # on ne coupe jamais la parole de Lily
    VOICE = "voix"              # toute parole soutenue l'interrompt
    WAKE = "eveil"              # seul « Lily » l'interrompt (robuste à l'écho)


@dataclass
class ListenerEvent:
    kind: str                          # wake | utterance | barge_in | stopped
    wake: WakeEvent | None = None
    utterance: Utterance | None = None


class VoiceListener:
    """Distribue les trames du micro selon le mode courant."""

    def __init__(
        self,
        mic: AudioInput,
        *,
        endpointer: Endpointer,
        on_event: Callable[[ListenerEvent], None],
        wake: WakeWordEngine | None = None,
        barge_in: BargeInDetector | None = None,
        barge_in_mode: BargeInMode | str = BargeInMode.VOICE,
        start_mode: ListenMode = ListenMode.PAUSED,
        corpus: Any = None,
    ) -> None:
        self.mic = mic
        self.endpointer = endpointer
        self.on_event = on_event
        self.wake = wake
        self.barge_in = barge_in
        self.barge_in_mode = BargeInMode(barge_in_mode)
        # Le corpus d'éveil, s'il est allumé : il ne voit que les trames du
        # mode « éveil » et la courte queue qui suit une détection.
        self.corpus = corpus

        self._mode = ListenMode(start_mode)
        self._demande: ListenMode | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._rechunk_eveil = Rechunker(wake.frame_size) if wake else None
        self._rechunk_vad = Rechunker(endpointer.vad.frame_size)

    # -- cycle de vie -------------------------------------------------------
    @property
    def mode(self) -> ListenMode:
        return self._mode

    @property
    def pending(self) -> ListenMode | None:
        """Le mode demandé mais pas encore appliqué (au plus une trame de retard)."""
        return self._demande

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._boucle, name="ecoute", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self.mic.stop()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def set_mode(self, mode: ListenMode) -> None:
        """Demande un changement de mode, appliqué à la trame suivante."""
        self._demande = ListenMode(mode)

    # -- boucle -------------------------------------------------------------
    def _boucle(self) -> None:
        try:
            self.mic.start()
        except Exception:
            logger.exception("Impossible d'ouvrir le micro ; l'écoute s'arrête.")
            self._emettre(ListenerEvent("stopped"))
            return
        try:
            for frame in self.mic.frames():
                if self._stop.is_set():
                    break
                if self._demande is not None:
                    self._appliquer(self._demande)
                    self._demande = None
                if self._mode is ListenMode.PAUSED:
                    continue
                if self._mode is ListenMode.WAKE:
                    self._traiter_eveil(frame)
                elif self._mode is ListenMode.UTTERANCE:
                    self._traiter_enonce(frame)
                elif self._mode is ListenMode.MONITOR:
                    self._traiter_surveillance(frame)
        except Exception:  # pragma: no cover - un micro qui lâche ne tue rien
            logger.exception("L'écoute s'est interrompue sur une erreur.")
        finally:
            try:
                self.mic.stop()
            except Exception:  # pragma: no cover
                logger.debug("Fermeture du micro en erreur.", exc_info=True)
            self._emettre(ListenerEvent("stopped"))

    def _appliquer(self, mode: ListenMode) -> None:
        logger.debug("écoute → %s", mode.value)
        self._mode = mode
        self._rechunk_vad.reset()
        if self._rechunk_eveil is not None:
            self._rechunk_eveil.reset()
        if mode is ListenMode.WAKE and self.wake is not None:
            self.wake.reset()
        elif mode is ListenMode.UTTERANCE:
            self.endpointer.reset()
        elif mode is ListenMode.MONITOR:
            if self.barge_in is not None:
                self.barge_in.reset()
            if self.barge_in_mode is BargeInMode.WAKE and self.wake is not None:
                self.wake.reset()

    def _emettre(self, evenement: ListenerEvent) -> None:
        try:
            self.on_event(evenement)
        except Exception:  # pragma: no cover - le consommateur ne casse pas l'écoute
            logger.exception("Consommateur d'événements en échec.")

    # -- traitements par mode ----------------------------------------------
    def _traiter_eveil(self, frame: bytes) -> None:
        if self.wake is None or self._rechunk_eveil is None:
            return
        if self.corpus is not None:
            self.corpus.alimenter(frame)
        for tranche in self._rechunk_eveil.push(frame):
            try:
                evenement = self.wake.process(tranche)
            except Exception:
                logger.exception("Le détecteur d'éveil a échoué ; écoute suspendue.")
                self._mode = ListenMode.PAUSED
                return
            if evenement is not None:
                # On se met en pause : c'est au consommateur de décider de la
                # suite. Sans cela, la même détection ressortirait en rafale.
                self._mode = ListenMode.PAUSED
                if self.corpus is not None:
                    self.corpus.declencher(evenement.score)
                self._emettre(ListenerEvent("wake", wake=evenement))
                return

    def _traiter_enonce(self, frame: bytes) -> None:
        if self.corpus is not None:
            # `completer`, pas `alimenter` : la fin du mot d'éveil est encore
            # devant nous, mais ce que dit l'utilisateur ne doit jamais
            # entrer dans le corpus.
            self.corpus.completer(frame)
        for tranche in self._rechunk_vad.push(frame):
            try:
                enonce = self.endpointer.push(tranche)
            except Exception:
                logger.exception("Le découpage d'énoncé a échoué ; écoute suspendue.")
                self._mode = ListenMode.PAUSED
                return
            # `None` = pas encore fini ; un énoncé faux = fini mais vide.
            if enonce is not None:
                self._mode = ListenMode.PAUSED
                self._emettre(ListenerEvent("utterance", utterance=enonce))
                return

    def _traiter_surveillance(self, frame: bytes) -> None:
        if self.barge_in_mode is BargeInMode.OFF:
            return
        if self.barge_in_mode is BargeInMode.WAKE:
            if self.wake is None or self._rechunk_eveil is None:
                return
            for tranche in self._rechunk_eveil.push(frame):
                if self.wake.process(tranche) is not None:
                    self._mode = ListenMode.PAUSED
                    self._emettre(ListenerEvent("barge_in"))
                    return
            return
        if self.barge_in is None:
            return
        for tranche in self._rechunk_vad.push(frame):
            if self.barge_in.push(tranche):
                self._mode = ListenMode.PAUSED
                self._emettre(ListenerEvent("barge_in"))
                return

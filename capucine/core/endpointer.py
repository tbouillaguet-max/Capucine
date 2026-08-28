"""Fin d'énoncé : savoir quand l'utilisateur a terminé, sans le couper.

C'est la pièce délicate de l'écoute. Trois pièges, trois réponses :

* **Couper trop tôt.** Une hésitation au milieu d'une phrase ressemble à une
  fin. D'où un temps de silence exigé (``silence_ms``), plus long que ce
  qu'on croit nécessaire — 700 ms par défaut.
* **Perdre le début du mot.** Le temps que le VAD s'accorde sur « il parle »,
  la première syllabe est déjà passée. D'où le **pré-roll** : les trames qui
  précèdent la détection sont conservées et remises en tête de l'énoncé.
* **Prendre une porte qui claque pour une phrase.** D'où une durée minimale
  de parole exigée avant de considérer qu'un énoncé a commencé, et une durée
  totale minimale avant de le transmettre.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from .audio import AudioBuffer
from .interfaces.vad import VADEngine
from .logging import get_logger

logger = get_logger("enonce")


class EndReason(StrEnum):
    SILENCE = "silence"          # fin normale : l'utilisateur s'est tu
    TOO_LONG = "trop_long"       # durée maximale atteinte
    TIMEOUT = "expire"           # personne n'a parlé
    TOO_SHORT = "trop_court"     # un bruit, pas une phrase


@dataclass
class Utterance:
    """Le résultat d'un découpage.

    Attention à la nuance : ``push()`` rend ``None`` tant que l'énoncé n'est
    pas terminé, et un ``Utterance`` **falsy** quand il s'est terminé sans
    rien contenir d'exploitable (personne n'a parlé, ou seulement un bruit).
    Côté appelant, on teste donc ``is not None`` pour « c'est fini » et la
    valeur de vérité pour « il y a quelque chose à transcrire ».
    """

    audio: AudioBuffer
    reason: EndReason
    speech_ms: float = 0.0

    def __bool__(self) -> bool:
        return self.reason in (EndReason.SILENCE, EndReason.TOO_LONG) and bool(self.audio)


class Endpointer:
    """Machine à états qui découpe un énoncé dans un flux de trames."""

    def __init__(
        self,
        vad: VADEngine,
        *,
        threshold: float = 0.5,
        min_speech_ms: float = 200.0,
        silence_ms: float = 700.0,
        pre_roll_ms: float = 300.0,
        max_utterance_s: float = 20.0,
        max_wait_s: float = 8.0,
        min_total_speech_ms: float = 300.0,
    ) -> None:
        self.vad = vad
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.silence_ms = silence_ms
        self.pre_roll_ms = pre_roll_ms
        self.max_utterance_s = max_utterance_s
        self.max_wait_s = max_wait_s
        self.min_total_speech_ms = min_total_speech_ms

        self.frame_ms = vad.frame_size / vad.sample_rate * 1000.0
        # Le pré-roll doit couvrir ce qui précède la détection **et** les
        # trames qui l'ont déclenchée, sinon on perd la première syllabe.
        profondeur = max(1, round((pre_roll_ms + min_speech_ms) / self.frame_ms))
        self._pre_roll: deque[bytes] = deque(maxlen=profondeur)
        self._frames: list[bytes] = []
        self._commence = False
        self._trames_parole = 0
        self._trames_silence = 0
        self._trames_parole_total = 0
        self._trames_ecoulees = 0

    # -- état ---------------------------------------------------------------
    @property
    def speech_started(self) -> bool:
        return self._commence

    def reset(self) -> None:
        self.vad.reset()
        self._pre_roll.clear()
        self._frames = []
        self._commence = False
        self._trames_parole = 0
        self._trames_silence = 0
        self._trames_parole_total = 0
        self._trames_ecoulees = 0

    # -- consommation -------------------------------------------------------
    def push(self, frame: bytes) -> Utterance | None:
        """Consomme une trame de la taille exigée par le VAD.

        Retourne l'énoncé dès qu'il est terminé, ``None`` sinon.
        """
        self._trames_ecoulees += 1
        probabilite = self.vad.speech_probability(frame)
        parle = probabilite >= self.threshold

        if not self._commence:
            self._pre_roll.append(frame)
            self._trames_parole = self._trames_parole + 1 if parle else 0
            if self._trames_parole * self.frame_ms >= self.min_speech_ms:
                self._commence = True
                # Le pré-roll contient déjà les trames déclenchantes.
                self._frames = list(self._pre_roll)
                self._trames_parole_total = self._trames_parole
                self._trames_silence = 0
                logger.debug("Début de parole (%.2f).", probabilite)
            elif self._trames_ecoulees * self.frame_ms >= self.max_wait_s * 1000:
                return self._terminer(EndReason.TIMEOUT)
            return None

        self._frames.append(frame)
        if parle:
            self._trames_parole_total += 1
            self._trames_silence = 0
        else:
            self._trames_silence += 1
            if self._trames_silence * self.frame_ms >= self.silence_ms:
                return self._terminer(EndReason.SILENCE)

        if len(self._frames) * self.frame_ms >= self.max_utterance_s * 1000:
            return self._terminer(EndReason.TOO_LONG)
        return None

    def _terminer(self, raison: EndReason) -> Utterance:
        parole_ms = self._trames_parole_total * self.frame_ms
        pcm = b"".join(self._frames)
        if raison is EndReason.TIMEOUT:
            audio = AudioBuffer(b"", self.vad.sample_rate)
        else:
            audio = AudioBuffer(pcm, self.vad.sample_rate)
            if parole_ms < self.min_total_speech_ms:
                # Une porte qui claque, une chaise qui grince : pas une phrase.
                logger.debug("Énoncé écarté : %.0f ms de parole seulement.", parole_ms)
                raison = EndReason.TOO_SHORT
        self.reset()
        return Utterance(audio=audio, reason=raison, speech_ms=parole_ms)

    def flush(self) -> Utterance:
        """Termine de force l'énoncé en cours (touche pressée, arrêt demandé)."""
        if not self._commence:
            self.reset()
            return Utterance(AudioBuffer(b"", self.vad.sample_rate), EndReason.TIMEOUT)
        return self._terminer(EndReason.SILENCE)


class BargeInDetector:
    """Repère que l'utilisateur reprend la parole pendant que Capucine répond.

    Le micro entend le haut-parleur : sans annulation d'écho, Capucine se
    couperait elle-même. Trois garde-fous, tous réglables :

    * un **seuil plus haut** que pour l'écoute normale ;
    * un **délai de garde** au début de la réponse, le temps que le niveau
      s'établisse ;
    * une **durée de parole soutenue** exigée, pour qu'un claquement ne
      suffise pas.

    Au casque, tout cela peut être abaissé. Sur haut-parleur, le mode
    « éveil » — n'interrompre que si l'on redit « Capucine » — reste le plus
    sûr.
    """

    def __init__(
        self,
        vad: VADEngine,
        *,
        threshold: float = 0.85,
        min_speech_ms: float = 300.0,
        guard_ms: float = 400.0,
    ) -> None:
        self.vad = vad
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.guard_ms = guard_ms
        self.frame_ms = vad.frame_size / vad.sample_rate * 1000.0
        self._trames_parole = 0
        self._trames_ecoulees = 0

    def reset(self) -> None:
        self.vad.reset()
        self._trames_parole = 0
        self._trames_ecoulees = 0

    def push(self, frame: bytes) -> bool:
        """``True`` dès que l'utilisateur a manifestement repris la parole."""
        self._trames_ecoulees += 1
        if self._trames_ecoulees * self.frame_ms <= self.guard_ms:
            return False
        if self.vad.speech_probability(frame) >= self.threshold:
            self._trames_parole += 1
        else:
            self._trames_parole = 0
        return self._trames_parole * self.frame_ms >= self.min_speech_ms

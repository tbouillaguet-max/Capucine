"""Interface de transcription.

L'entrée est un ``AudioBuffer`` (PCM 16 bits mono), pas un tableau numpy :
c'est ce qui circule depuis le micro, et c'est au moteur — qui dépend de
numpy de toute façon — de faire la conversion s'il en a besoin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..audio import AudioBuffer


@dataclass
class Transcription:
    text: str
    language: str = "fr"
    duration_s: float = 0.0
    confidence: float | None = None

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class STTEngine(ABC):
    """Contrat minimal d'un moteur de reconnaissance vocale local.

    Changer de moteur — ``faster-whisper`` sur PC, Vosk sur Raspberry Pi — est
    une ligne de configuration. Un moteur ``whisper.cpp`` se brancherait ici
    sans que le pipeline en sache rien.
    """

    name: str = "stt"

    @abstractmethod
    def available(self) -> bool:
        """Ne lève jamais : un moteur absent est un état, pas une erreur."""

    @abstractmethod
    def transcribe(self, audio: AudioBuffer) -> Transcription:
        """Transcrit un extrait audio capté."""

    def warmup(self) -> None:
        """Charge le modèle et fait une passe à vide, pour ne pas payer le
        chargement devant l'utilisateur."""

    def close(self) -> None:
        ...

    def describe(self) -> str:
        return self.name

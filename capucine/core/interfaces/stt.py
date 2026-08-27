"""Interface de transcription."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typage seul, aucun import à l'exécution
    import numpy as np


@dataclass
class Transcription:
    text: str
    language: str = "fr"
    duration_s: float = 0.0
    confidence: float | None = None

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class STTEngine(ABC):
    """Contrat minimal d'un moteur de reconnaissance vocale local."""

    name: str = "stt"

    @abstractmethod
    def available(self) -> bool:
        """Ne lève jamais."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray | Any, sample_rate: int) -> Transcription:
        """Transcrit un extrait audio.

        Args:
            audio: PCM mono en float32, valeurs dans [-1, 1].
            sample_rate: Fréquence d'échantillonnage de ``audio``.
        """

    def warmup(self) -> None:
        """Charge le modèle et fait une passe à vide."""

    def close(self) -> None:
        ...

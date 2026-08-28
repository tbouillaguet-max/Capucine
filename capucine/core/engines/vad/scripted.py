"""VAD scripté : la suite de probabilités est écrite d'avance.

Il rend les tests de la machine à états d'énoncé lisibles — « parle, parle,
silence, silence, silence » — au lieu de fabriquer un signal audio plausible.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ...interfaces.vad import VADEngine


class ScriptedVAD(VADEngine):
    name = "scripted"

    def __init__(
        self,
        probabilities: Iterable[float] | None = None,
        default: float = 0.0,
        sample_rate: int = 16000,
        frame_size: int = 512,
        **_ignored: Any,
    ) -> None:
        self.probabilities = list(probabilities or ())
        self.default = default
        self._sample_rate = sample_rate
        self._frame_size = frame_size
        self.index = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return self._frame_size

    def available(self) -> bool:
        return True

    def reset(self) -> None:
        self.index = 0

    def speech_probability(self, frame: bytes) -> float:
        if self.index < len(self.probabilities):
            valeur = self.probabilities[self.index]
        else:
            valeur = self.default
        self.index += 1
        return valeur

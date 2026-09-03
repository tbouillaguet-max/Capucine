"""Mot d'éveil scripté, pour les tests et les démonstrations.

``ScriptedWakeWord([3, 10])`` se déclenche à la quatrième et à la onzième
trame. C'est tout ce dont on a besoin pour éprouver la boucle d'éveil sans
prononcer un mot.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from ...interfaces.wake import WakeEvent, WakeWordEngine


class ScriptedWakeWord(WakeWordEngine):
    name = "scripted"

    def __init__(
        self,
        hits: Iterable[int] | None = None,
        word: str = "lily",
        sample_rate: int = 16000,
        frame_size: int = 1280,
        **_ignored: Any,
    ) -> None:
        self.hits = set(hits or ())
        self.word = word
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

    def process(self, frame: bytes) -> WakeEvent | None:
        index = self.index
        self.index += 1
        if index in self.hits:
            return WakeEvent(word=self.word, score=1.0, timestamp=time.monotonic())
        return None

    def reset(self) -> None:
        pass

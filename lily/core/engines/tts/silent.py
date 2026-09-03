"""Synthèse factice : du silence de durée réaliste, une phrase par morceau.

Elle sert à dérouler tout le chemin vocal — découpage en phrases, streaming,
interruption entre deux phrases, écriture d'un WAV — sans voix installée. Les
morceaux ont une durée proportionnelle au texte, pour que les mesures de
latence et les scénarios de barge-in restent réalistes.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from ...audio import SAMPLE_WIDTH, AudioChunk
from ...interfaces.tts import TTSEngine
from ...text import split_sentences


class SilentTTS(TTSEngine):
    name = "silent"

    def __init__(
        self,
        sample_rate: int = 22050,
        ms_per_char: float = 60.0,
        sentence_min_chars: int = 1,
        **_ignored: Any,
    ) -> None:
        self.sample_rate = sample_rate
        self.ms_per_char = ms_per_char
        self.sentence_min_chars = sentence_min_chars
        self.spoken: list[str] = []

    def available(self) -> bool:
        return True

    def synthesize(self, text: str, cancel: threading.Event | None = None) -> Iterator[AudioChunk]:
        for phrase in split_sentences(text, min_chars=self.sentence_min_chars):
            if cancel is not None and cancel.is_set():
                return
            echantillons = int(self.sample_rate * len(phrase) * self.ms_per_char / 1000)
            self.spoken.append(phrase)
            yield AudioChunk(
                pcm=b"\x00" * (echantillons * SAMPLE_WIDTH),
                sample_rate=self.sample_rate,
                text=phrase,
            )

    def describe(self) -> str:
        return "silent"

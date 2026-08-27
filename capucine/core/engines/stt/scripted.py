"""Transcription factice : rend des phrases prévues d'avance.

Elle permet de dérouler le pipeline vocal complet — capture, transcription,
réflexion, synthèse, lecture — sans micro et sans modèle, en intégration
continue comme en développement (``--stt scripted``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

from ...audio import AudioBuffer
from ...interfaces.stt import STTEngine, Transcription


class ScriptedSTT(STTEngine):
    name = "scripted"

    def __init__(self, transcriptions: Iterable[str] | None = None, language: str = "fr", **_ignored: Any) -> None:
        self.transcriptions: deque[str] = deque(transcriptions or ())
        self.language = language
        self.buffers: list[AudioBuffer] = []

    def push(self, *textes: str) -> ScriptedSTT:
        self.transcriptions.extend(textes)
        return self

    def available(self) -> bool:
        return True

    def transcribe(self, audio: AudioBuffer) -> Transcription:
        self.buffers.append(audio)
        texte = self.transcriptions.popleft() if self.transcriptions else ""
        return Transcription(texte, self.language, audio.duration_s, confidence=1.0)

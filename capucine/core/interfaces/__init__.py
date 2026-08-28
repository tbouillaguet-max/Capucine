"""Interfaces abstraites des quatre étages : éveil, STT, LLM, TTS.

Ce paquet ne contient **que** des ABC et des types de données. Il ne doit
jamais importer une dépendance lourde : importer ``STTEngine`` ne doit pas
tirer ``faster-whisper`` ni ``torch``. Les implémentations concrètes vivent
dans ``capucine.core.engines`` et sont importées paresseusement, d'après la
configuration.
"""

from ..audio import AudioBuffer, AudioChunk
from .llm import LLMEngine, Message, ToolCall
from .stt import STTEngine, Transcription
from .tts import TTSEngine
from .vad import VADEngine
from .wake import WakeEvent, WakeWordEngine

__all__ = [
    "AudioBuffer",
    "AudioChunk",
    "LLMEngine",
    "Message",
    "ToolCall",
    "STTEngine",
    "Transcription",
    "TTSEngine",
    "VADEngine",
    "WakeWordEngine",
    "WakeEvent",
]

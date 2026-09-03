"""Transcription par Vosk.

C'est le repli du Raspberry Pi : beaucoup plus léger que Whisper, moins précis,
mais il consomme directement le PCM 16 bits 16 kHz que produit le micro — sans
numpy, sans conversion. Il sert aussi de repli au mot d'éveil à l'étape 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...audio import AudioBuffer
from ...errors import EngineUnavailable
from ...interfaces.stt import STTEngine, Transcription
from ...logging import get_logger

logger = get_logger("stt.vosk")


class VoskSTT(STTEngine):
    name = "vosk"

    def __init__(
        self,
        model_path: str | Path = "models/vosk/vosk-model-small-fr-0.22",
        sample_rate: int = 16000,
        language: str = "fr",
        log_level: int = -1,
        **_ignored: Any,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.sample_rate = sample_rate
        self.language = language
        self.log_level = log_level
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                import vosk
            except ImportError as exc:  # pragma: no cover
                raise EngineUnavailable(
                    "Le paquet « vosk » est absent. Installez-le avec : pip install vosk"
                ) from exc
            if not self.model_path.is_dir():
                raise EngineUnavailable(
                    f"Modèle Vosk introuvable : {self.model_path}. "
                    "Téléchargez-en un sur https://alphacephei.com/vosk/models "
                    "(vosk-model-small-fr-0.22 pour le français) et décompressez-le là."
                )
            vosk.SetLogLevel(self.log_level)
            logger.info("Chargement du modèle Vosk « %s »", self.model_path.name)
            self._model = vosk.Model(str(self.model_path))
        return self._model

    def available(self) -> bool:
        try:
            import vosk  # noqa: F401
        except ImportError:
            return False
        return self.model_path.is_dir()

    def transcribe(self, audio: AudioBuffer) -> Transcription:
        if not audio.pcm:
            return Transcription("", self.language, 0.0)
        if audio.sample_rate != self.sample_rate:
            raise EngineUnavailable(
                f"Vosk attend du {self.sample_rate} Hz, reçu {audio.sample_rate} Hz. "
                "Alignez audio.sample_rate sur stt.sample_rate."
            )
        import vosk

        recogniser = vosk.KaldiRecognizer(self._get_model(), self.sample_rate)
        recogniser.SetWords(True)
        recogniser.AcceptWaveform(audio.pcm)
        resultat = json.loads(recogniser.FinalResult())

        mots = resultat.get("result") or []
        confiance = (
            round(sum(mot.get("conf", 0.0) for mot in mots) / len(mots), 3) if mots else None
        )
        return Transcription(
            text=(resultat.get("text") or "").strip(),
            language=self.language,
            duration_s=audio.duration_s,
            confidence=confiance,
        )

    def warmup(self) -> None:
        try:
            self._get_model()
        except EngineUnavailable as exc:  # pragma: no cover
            logger.warning("Préchauffage Vosk impossible : %s", exc)

    def close(self) -> None:
        self._model = None

    def describe(self) -> str:
        return f"vosk:{self.model_path.name}"

"""Synthèse vocale par Piper, en local.

Piper produit **un morceau audio par phrase**, ce qui tombe bien : c'est
exactement l'unité de streaming dont le pipeline a besoin pour commencer à
parler avant la fin de l'inférence du modèle de langage, et pour s'interrompre
proprement quand l'utilisateur coupe la parole.

On découpe nous-mêmes le texte en phrases plutôt que de laisser Piper le faire,
pour trois raisons : chaque morceau rendu porte le texte qui lui correspond
(journal et diagnostic), ``cancel`` est consulté entre chaque phrase, et la
latence de la première phrase est mesurable séparément.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ...audio import AudioChunk
from ...errors import EngineUnavailable
from ...interfaces.tts import TTSEngine
from ...logging import get_logger
from ...text import split_sentences

logger = get_logger("tts.piper")

VOIX_PAR_DEFAUT = "fr_FR-siwis-medium"


class PiperTTS(TTSEngine):
    name = "piper"

    def __init__(
        self,
        voice: str = VOIX_PAR_DEFAUT,
        models_dir: str | Path = "models/piper",
        speed: float = 1.0,
        volume: float = 1.0,
        noise_scale: float | None = None,
        noise_w_scale: float | None = None,
        speaker_id: int | None = None,
        use_cuda: bool = False,
        sentence_min_chars: int = 1,
        **_ignored: Any,
    ) -> None:
        self.voice = voice
        self.models_dir = Path(models_dir).expanduser()
        self.speed = speed if speed > 0 else 1.0
        self.volume = volume
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.speaker_id = speaker_id
        self.use_cuda = use_cuda
        self.sentence_min_chars = sentence_min_chars
        self._voice: Any = None
        self._config: Any = None

    # -- ressources ---------------------------------------------------------
    def voice_path(self) -> Path:
        return self.models_dir / f"{self.voice}.onnx"

    def available(self) -> bool:
        try:
            import piper  # noqa: F401
        except ImportError:
            return False
        return self.voice_path().exists()

    def _get_voice(self) -> Any:
        if self._voice is not None:
            return self._voice
        try:
            from piper import PiperVoice, SynthesisConfig
        except ImportError as exc:  # pragma: no cover - dépend de l'install
            raise EngineUnavailable(
                "Le paquet « piper-tts » est absent. Installez-le avec : pip install piper-tts"
            ) from exc

        chemin = self.voice_path()
        if not chemin.exists():
            raise EngineUnavailable(
                f"Voix Piper introuvable : {chemin}. Téléchargez-la avec :\n"
                f"  python -m lily.core.downloads voix {self.voice}\n"
                f"ou : python -m piper.download_voices {self.voice} "
                f"--download-dir {self.models_dir}"
            )
        logger.info("Chargement de la voix Piper « %s »", self.voice)
        self._voice = PiperVoice.load(chemin, use_cuda=self.use_cuda)
        self._config = SynthesisConfig(
            speaker_id=self.speaker_id,
            # Piper raisonne en durée : plus le facteur est grand, plus c'est
            # lent. On expose une vitesse, qui est l'inverse.
            length_scale=1.0 / self.speed if self.speed != 1.0 else None,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
            volume=self.volume,
        )
        return self._voice

    # -- synthèse -----------------------------------------------------------
    def synthesize(self, text: str, cancel: threading.Event | None = None) -> Iterator[AudioChunk]:
        phrases = split_sentences(text, min_chars=self.sentence_min_chars)
        if not phrases:
            return
        voix = self._get_voice()
        for index, phrase in enumerate(phrases):
            if cancel is not None and cancel.is_set():
                logger.debug("Synthèse interrompue avant la phrase %d.", index + 1)
                return
            depart = time.perf_counter()
            morceaux = list(voix.synthesize(phrase, syn_config=self._config))
            if not morceaux:
                continue
            pcm = b"".join(morceau.audio_int16_bytes for morceau in morceaux)
            chunk = AudioChunk(pcm=pcm, sample_rate=morceaux[0].sample_rate, text=phrase)
            logger.debug(
                "Phrase %d synthétisée en %.0f ms pour %.2f s d'audio",
                index + 1, (time.perf_counter() - depart) * 1000, chunk.duration_s,
            )
            yield chunk

    def warmup(self) -> None:
        try:
            list(self.synthesize("Bonjour."))
        except Exception as exc:  # pragma: no cover - dépend de l'install
            logger.warning("Préchauffage Piper impossible : %s", exc)

    def close(self) -> None:
        self._voice = None
        self._config = None

    def describe(self) -> str:
        return f"piper:{self.voice}"

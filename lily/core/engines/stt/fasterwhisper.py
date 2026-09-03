"""Transcription par ``faster-whisper`` (Whisper compilé avec CTranslate2).

Deux précautions valent d'être expliquées, parce qu'elles ne se devinent pas :

* **Whisper hallucine sur le silence.** Un extrait vide ou très faible produit
  régulièrement des phrases entières et plausibles (« Sous-titrage… »,
  « Merci d'avoir regardé cette vidéo »). Avec un assistant déclenché à la
  voix, cela se traduit par des commandes fantômes. On coupe donc court avant
  le modèle quand le niveau sonore est trop bas, et on écarte une courte liste
  de formules connues.
* **``condition_on_previous_text=False``.** Sur des énoncés courts et
  indépendants, conditionner sur le tour précédent fait boucler le modèle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...audio import AudioBuffer
from ...errors import EngineUnavailable
from ...interfaces.stt import STTEngine, Transcription
from ...logging import get_logger
from ...text import normalize

logger = get_logger("stt.whisper")

# Formules que Whisper produit sur du silence ou du souffle, en français.
HALLUCINATIONS = (
    "sous titrage societe radio canada",
    "sous titres realises par",
    "merci d avoir regarde cette video",
    "merci a tous et a bientot",
    "abonnez vous",
    "amara org",
    "c est la fin de la video",
)


class FasterWhisperSTT(STTEngine):
    name = "faster-whisper"

    def __init__(
        self,
        model: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "fr",
        beam_size: int = 5,
        cpu_threads: int = 0,
        download_root: str | Path | None = None,
        initial_prompt: str | None = None,
        vad_filter: bool = False,
        no_speech_threshold: float = 0.6,
        min_rms: float = 0.006,
        min_duration_s: float = 0.25,
        drop_hallucinations: bool = True,
        **_ignored: Any,
    ) -> None:
        self.model_name = model
        self.device = device
        # « auto » est notre vocabulaire ; CTranslate2 dit « default ».
        self.compute_type = "default" if compute_type in ("auto", "") else compute_type
        self.language = language
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.download_root = str(download_root) if download_root else None
        self.initial_prompt = initial_prompt
        self.vad_filter = vad_filter
        self.no_speech_threshold = no_speech_threshold
        self.min_rms = min_rms
        self.min_duration_s = min_duration_s
        self.drop_hallucinations = drop_hallucinations
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - dépend de l'install
                raise EngineUnavailable(
                    "Le paquet « faster-whisper » est absent. "
                    "Installez-le avec : pip install faster-whisper"
                ) from exc
            logger.info(
                "Chargement du modèle Whisper « %s » (%s, %s)",
                self.model_name, self.device, self.compute_type,
            )
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                download_root=self.download_root,
            )
        return self._model

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def transcribe(self, audio: AudioBuffer) -> Transcription:
        if audio.duration_s < self.min_duration_s:
            logger.debug("Extrait trop court (%.2f s), ignoré.", audio.duration_s)
            return Transcription("", self.language, audio.duration_s)

        niveau = audio.rms()
        if niveau < self.min_rms:
            # On n'envoie même pas l'extrait au modèle : c'est du silence, et
            # Whisper y verrait une phrase.
            logger.debug("Niveau sonore trop bas (%.4f), extrait ignoré.", niveau)
            return Transcription("", self.language, audio.duration_s)

        segments, info = self._get_model().transcribe(
            audio.to_float32(),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            initial_prompt=self.initial_prompt,
            no_speech_threshold=self.no_speech_threshold,
            condition_on_previous_text=False,
        )
        segments = list(segments)  # le générateur pilote réellement le décodage
        texte = " ".join(segment.text.strip() for segment in segments).strip()

        if self.drop_hallucinations and _est_une_hallucination(texte):
            logger.info("Transcription écartée (hallucination probable) : %r", texte)
            return Transcription("", self.language, audio.duration_s)

        confiance = None
        if segments:
            import math

            moyenne = sum(s.avg_logprob for s in segments) / len(segments)
            confiance = round(math.exp(moyenne), 3)

        return Transcription(
            text=texte,
            language=getattr(info, "language", self.language) or self.language,
            duration_s=audio.duration_s,
            confidence=confiance,
        )

    def warmup(self) -> None:
        try:
            modele = self._get_model()
            silence = AudioBuffer(b"\x00\x00" * 16000, 16000)
            list(modele.transcribe(silence.to_float32(), language=self.language, beam_size=1)[0])
        except Exception as exc:  # pragma: no cover - dépend de l'install
            logger.warning("Préchauffage Whisper impossible : %s", exc)

    def close(self) -> None:
        self._model = None

    def describe(self) -> str:
        return f"faster-whisper:{self.model_name}"


def _est_une_hallucination(texte: str) -> bool:
    if not texte:
        return False
    normalise = normalize(texte)
    return any(motif in normalise for motif in HALLUCINATIONS)

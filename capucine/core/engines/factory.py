"""Fabrique de moteurs : passer d'un backend à l'autre est une ligne de config.

Les modules concrets sont importés **à l'instanciation seulement**. C'est ce
qui permet à ``python main.py --text`` de démarrer en une seconde sur un
Raspberry Pi sans charger torch, et aux tests de tourner sans la stack audio.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

from ..audio import (
    AudioInput,
    AudioOutput,
    MemoryAudioOutput,
    NullAudioOutput,
    SoundDeviceInput,
    SoundDeviceOutput,
    WavFileInput,
    WavFileOutput,
)
from ..errors import ConfigError, EngineUnavailable
from ..interfaces.llm import LLMEngine
from ..interfaces.stt import STTEngine
from ..interfaces.tts import TTSEngine
from ..interfaces.vad import VADEngine
from ..interfaces.wake import WakeWordEngine
from ..logging import get_logger

logger = get_logger("engines")

# nom de config -> (module, classe)
LLM_ENGINES: dict[str, tuple[str, str]] = {
    "mock": ("capucine.core.engines.llm.mock", "MockLLM"),
    "ollama": ("capucine.core.engines.llm.ollama", "OllamaLLM"),
    "llamacpp": ("capucine.core.engines.llm.llamacpp", "LlamaCppLLM"),
}

STT_ENGINES: dict[str, tuple[str, str]] = {
    "faster-whisper": ("capucine.core.engines.stt.fasterwhisper", "FasterWhisperSTT"),
    "whisper": ("capucine.core.engines.stt.fasterwhisper", "FasterWhisperSTT"),
    "vosk": ("capucine.core.engines.stt.vosk", "VoskSTT"),
    "scripted": ("capucine.core.engines.stt.scripted", "ScriptedSTT"),
}

TTS_ENGINES: dict[str, tuple[str, str]] = {
    "piper": ("capucine.core.engines.tts.piper", "PiperTTS"),
    "silent": ("capucine.core.engines.tts.silent", "SilentTTS"),
}

VAD_ENGINES: dict[str, tuple[str, str]] = {
    "silero": ("capucine.core.engines.vad.silero", "SileroVAD"),
    "energie": ("capucine.core.engines.vad.energy", "EnergyVAD"),
    "energy": ("capucine.core.engines.vad.energy", "EnergyVAD"),
    "scripted": ("capucine.core.engines.vad.scripted", "ScriptedVAD"),
}

WAKE_ENGINES: dict[str, tuple[str, str]] = {
    "openwakeword": ("capucine.core.engines.wake.openwakeword", "OpenWakeWordEngine"),
    "vosk": ("capucine.core.engines.wake.vosk", "VoskWakeWord"),
    "scripted": ("capucine.core.engines.wake.scripted", "ScriptedWakeWord"),
}


def _instantiate(table: Mapping[str, tuple[str, str]], kind: str, name: str, options: Mapping[str, Any]) -> Any:
    if name not in table:
        known = ", ".join(sorted(table)) or "aucun pour l'instant"
        raise ConfigError(f"Moteur {kind} inconnu : « {name} ». Disponibles : {known}.")
    module_name, class_name = table[name]
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise EngineUnavailable(
            f"Le moteur {kind} « {name} » n'est pas installable en l'état : {exc}"
        ) from exc
    factory = getattr(module, class_name)
    return factory(**dict(options))


def build_llm(config: Any) -> LLMEngine:
    """Construit le moteur LLM décrit par ``[llm]``, avec repli sur le factice.

    Le repli est volontaire : une Capucine sans modèle doit continuer à
    exécuter des compétences plutôt que refuser de démarrer.
    """
    section = dict(config.section("llm"))
    name = str(section.pop("engine", "mock"))
    section.pop("router", None)
    try:
        engine = _instantiate(LLM_ENGINES, "LLM", name, section)
    except (ConfigError, EngineUnavailable) as exc:
        logger.error("%s", exc)
        logger.warning("Repli sur le moteur factice : les compétences resteront utilisables.")
        return _instantiate(LLM_ENGINES, "LLM", "mock", {})

    if not engine.available():
        logger.warning(
            "Moteur LLM « %s » injoignable. Repli sur le moteur factice : "
            "les compétences restent utilisables, la conversation libre non.",
            engine.describe(),
        )
        return _instantiate(LLM_ENGINES, "LLM", "mock", {})
    logger.info("Moteur LLM : %s", engine.describe())
    return engine


def build_stt(config: Any) -> STTEngine:
    """Construit le moteur de transcription décrit par ``[stt]``.

    Aucun repli silencieux, contrairement au LLM : une Capucine qui n'entend
    pas ne sert à rien, et faire semblant serait pire que refuser. Le message
    d'erreur nomme ce qu'il faut installer ou télécharger.
    """
    section = dict(config.section("stt"))
    name = str(section.pop("engine", "faster-whisper"))
    section.pop("models_dir", None)
    if name in ("faster-whisper", "whisper"):
        section.setdefault("download_root", str(config.resolve_path("stt.models_dir") or ""))
        if not section["download_root"]:
            section.pop("download_root")
    engine = _instantiate(STT_ENGINES, "STT", name, section)
    if not engine.available():
        raise EngineUnavailable(
            f"Moteur de transcription « {engine.describe()} » indisponible. "
            "Installez la chaîne audio (pip install -e \".[audio]\") et téléchargez le "
            "modèle (python -m capucine.core.downloads tout), ou utilisez --text."
        )
    logger.info("Moteur STT : %s", engine.describe())
    return engine


def build_tts(config: Any) -> TTSEngine | None:
    """Construit le moteur de synthèse décrit par ``[tts]``.

    Ici le repli est légitime : sans voix, Capucine affiche sa réponse au lieu
    de la dire. C'est dégradé mais utilisable, alors qu'une oreille absente ne
    l'est pas.
    """
    section = dict(config.section("tts"))
    name = str(section.pop("engine", "piper"))
    dossier = config.resolve_path("tts.models_dir")
    if dossier is not None:
        section["models_dir"] = str(dossier)
    try:
        engine = _instantiate(TTS_ENGINES, "TTS", name, section)
    except (ConfigError, EngineUnavailable) as exc:
        logger.error("%s", exc)
        return None
    if not engine.available():
        logger.warning(
            "Moteur de synthèse « %s » indisponible : Capucine affichera ses réponses "
            "au lieu de les dire. Téléchargez la voix avec : "
            "python -m capucine.core.downloads voix",
            engine.describe(),
        )
        return None
    logger.info("Moteur TTS : %s", engine.describe())
    return engine


def build_audio_input(config: Any, wav: str | None = None) -> AudioInput:
    """Micro réel, ou rejeu d'un fichier WAV — l'entrée « sans micro »."""
    section = config.section("audio")
    frame_ms = int(section.get("frame_ms", 30))
    if wav:
        logger.info("Entrée audio : rejeu de %s", wav)
        return WavFileInput(wav, frame_ms=frame_ms)
    entree = SoundDeviceInput(
        sample_rate=int(section.get("sample_rate", 16000)),
        frame_ms=frame_ms,
        device=section.get("input_device") or None,
    )
    if not entree.available():
        raise EngineUnavailable(
            "Aucun micro utilisable. Installez PortAudio (sous Linux : "
            "sudo apt install libportaudio2) et « pip install sounddevice », "
            "vérifiez la liste avec « python main.py --devices », ou utilisez --text."
        )
    return entree


def build_audio_output(
    config: Any, wav: str | None = None, silent: bool = False
) -> AudioOutput | None:
    """Haut-parleur réel, fichier WAV, ou rien.

    ``None`` signifie « pas de sortie audio du tout » : le pipeline affiche
    alors ses réponses au lieu de les dire. On le décide au démarrage plutôt
    que d'échouer à chaque phrase prononcée.
    """
    if silent:
        return NullAudioOutput()
    if wav:
        logger.info("Sortie audio : écriture dans %s", wav)
        return WavFileOutput(Path(wav))
    sortie = SoundDeviceOutput(device=config.section("audio").get("output_device") or None)
    if not sortie.available():
        logger.warning(
            "Aucun haut-parleur utilisable : Capucine affichera ses réponses au lieu "
            "de les dire. Installez PortAudio, ou vérifiez « python main.py --devices »."
        )
        return None
    return sortie


__all__ = [
    "LLM_ENGINES", "STT_ENGINES", "TTS_ENGINES", "VAD_ENGINES", "WAKE_ENGINES",
    "build_llm", "build_stt", "build_tts", "build_vad", "build_wake",
    "build_audio_input", "build_audio_output", "MemoryAudioOutput",
]


def build_vad(config: Any) -> VADEngine:
    """Construit le détecteur d'activité vocale décrit par ``[vad]``.

    Repli sur le VAD par énergie si Silero manque : sans détection de fin de
    phrase, la boucle vocale ne tourne pas du tout, alors qu'un VAD moins fin
    reste parfaitement utilisable au calme.
    """
    section = dict(config.section("vad"))
    name = str(section.pop("engine", "silero"))
    for cle in ("threshold", "silence_ms", "min_speech_ms", "pre_roll_ms",
                "max_utterance_s", "max_wait_s", "min_total_speech_ms"):
        section.pop(cle, None)
    dossier = config.resolve_path("vad.models_dir")
    if dossier is not None:
        section["models_dir"] = str(dossier)
    section.setdefault("sample_rate", int(config.get("audio.sample_rate", 16000)))

    try:
        engine = _instantiate(VAD_ENGINES, "VAD", name, section)
        if engine.available():
            logger.info("Moteur VAD : %s", engine.describe())
            return engine
        logger.warning("Moteur VAD « %s » indisponible.", engine.describe())
    except (ConfigError, EngineUnavailable) as exc:
        logger.warning("%s", exc)

    logger.warning(
        "Repli sur le VAD par énergie : moins fin dans le bruit, mais toujours "
        "opérationnel. Pour Silero : pip install --no-deps silero-vad onnxruntime"
    )
    return _instantiate(
        VAD_ENGINES, "VAD", "energie",
        {"sample_rate": section.get("sample_rate", 16000),
         "frame_size": int(int(config.get("audio.sample_rate", 16000)) * 0.03)},
    )


def build_wake(config: Any) -> WakeWordEngine | None:
    """Construit le détecteur de mot d'éveil décrit par ``[wake]``.

    Chaîne de replis voulue par le projet : le modèle openWakeWord
    « capucine » doit être entraîné, ce qui prend du temps ; tant qu'il
    n'existe pas, Vosk à grammaire restreinte prend le relais. Si aucun des
    deux n'est disponible, on rend ``None`` et Capucine écoute en permanence,
    sans mot d'éveil — dégradé, mais utilisable.
    """
    section = dict(config.section("wake"))
    name = str(section.pop("engine", "openwakeword"))
    chemin_vosk = section.pop("vosk_model_path", None)
    dossier = config.resolve_path("wake.models_dir")
    if dossier is not None:
        section["models_dir"] = str(dossier)

    def _essayer(nom: str, options: dict[str, Any]) -> WakeWordEngine | None:
        try:
            engine = _instantiate(WAKE_ENGINES, "éveil", nom, options)
        except (ConfigError, EngineUnavailable) as exc:
            logger.warning("%s", exc)
            return None
        if engine.available():
            logger.info("Mot d'éveil : %s", engine.describe())
            return engine
        logger.warning("Moteur d'éveil « %s » indisponible.", engine.describe())
        return None

    engine = _essayer(name, section)
    if engine is not None:
        return engine

    if name == "openwakeword":
        options = dict(section)
        options.pop("models_dir", None)
        options.pop("inference_framework", None)
        options.pop("threshold", None)
        if chemin_vosk:
            options["model_path"] = str(config.resolve_path("wake.vosk_model_path") or chemin_vosk)
        logger.warning(
            "Le modèle openWakeWord « %s » n'est pas entraîné : repli sur Vosk. "
            "Pour l'entraîner : python tools/entrainer_capucine.py --preparer",
            section.get("word", "capucine"),
        )
        engine = _essayer("vosk", options)
        if engine is not None:
            return engine

    logger.warning(
        "Aucun détecteur de mot d'éveil disponible : Capucine écoutera en "
        "permanence. Entraînez le modèle, installez Vosk, ou utilisez "
        "--push-to-talk."
    )
    return None

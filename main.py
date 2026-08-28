#!/usr/bin/env python3
"""Point d'entrée de Capucine.

    python main.py                        # écoute permanente : dites « Capucine »
    python main.py --text                 # boucle clavier, sans micro ni haut-parleur
    python main.py --text --llm mock      # sans aucun modèle de langage
    python main.py --push-to-talk         # [Entrée] pour parler, sans mot d'éveil
    python main.py --wav-in essai.wav     # rejoue un fichier : un tour, sans micro
    python main.py --devices              # inventaire des périphériques audio
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capucine.app import build_assistant, run_text_mode, run_voice_mode  # noqa: E402
from capucine.core.audio import list_devices  # noqa: E402
from capucine.core.config import load_config  # noqa: E402
from capucine.core.errors import CapucineError, EngineUnavailable  # noqa: E402
from capucine.core.logging import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="capucine",
        description="Capucine — assistante vocale locale, hors-ligne et extensible par fichiers.",
    )
    parser.add_argument("--text", action="store_true",
                        help="mode clavier : court-circuite micro et haut-parleur")
    parser.add_argument("--once", metavar="PHRASE",
                        help="traite une seule phrase puis quitte (implique --text)")
    parser.add_argument("--profile", choices=["pc", "pi"],
                        help="profil de configuration (détecté automatiquement sinon)")
    parser.add_argument("--config", metavar="FICHIER",
                        help="fichier TOML supplémentaire, appliqué par-dessus le profil")
    parser.add_argument("--llm", metavar="MOTEUR",
                        help="remplace llm.engine : ollama, llamacpp ou mock")
    parser.add_argument("--stt", metavar="MOTEUR",
                        help="remplace stt.engine : faster-whisper, vosk ou scripted")
    parser.add_argument("--tts", metavar="MOTEUR",
                        help="remplace tts.engine : piper ou silent")
    parser.add_argument("--wake", metavar="MOTEUR",
                        help="remplace wake.engine : openwakeword ou vosk")
    parser.add_argument("--vad", metavar="MOTEUR",
                        help="remplace vad.engine : silero ou energie")
    parser.add_argument("--plugins", metavar="DOSSIER", action="append",
                        help="dossier de plugins supplémentaire (répétable)")
    parser.add_argument("--no-hot-reload", action="store_true",
                        help="ne surveille pas plugins/ (la commande /recharge reste)")

    ecoute = parser.add_argument_group("écoute")
    ecoute.add_argument("--push-to-talk", action="store_true",
                        help="[Entrée] pour parler, au lieu du mot d'éveil")
    ecoute.add_argument("--no-wake", action="store_true",
                        help="écoute permanente sans mot d'éveil (VAD seul)")
    ecoute.add_argument("--barge-in", choices=["voix", "eveil", "off"],
                        help="ce qui autorise à couper la parole à Capucine")

    audio = parser.add_argument_group("audio")
    audio.add_argument("--devices", action="store_true",
                       help="liste les périphériques audio puis quitte")
    audio.add_argument("--wav-in", metavar="FICHIER",
                       help="rejoue un WAV 16 bits mono au lieu du micro, un seul tour")
    audio.add_argument("--wav-out", metavar="FICHIER",
                       help="écrit la parole dans un WAV au lieu du haut-parleur")
    audio.add_argument("--muet", action="store_true",
                       help="ne joue aucun son (mesure de latence sans lecture)")

    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--json-logs", action="store_true",
                        help="journal en JSON, une ligne par événement")
    return parser.parse_args(argv)


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.llm:
        overrides.setdefault("llm", {})["engine"] = args.llm
    if args.stt:
        overrides.setdefault("stt", {})["engine"] = args.stt
    if args.tts:
        overrides.setdefault("tts", {})["engine"] = args.tts
    if args.wake:
        overrides.setdefault("wake", {})["engine"] = args.wake
    if args.vad:
        overrides.setdefault("vad", {})["engine"] = args.vad
    if args.barge_in:
        overrides.setdefault("barge_in", {})["mode"] = args.barge_in
    if args.plugins:
        overrides.setdefault("plugins", {})["paths"] = list(args.plugins)
    if args.no_hot_reload:
        overrides.setdefault("plugins", {})["hot_reload"] = False
    if args.log_level:
        overrides.setdefault("logging", {})["level"] = args.log_level
    if args.json_logs:
        overrides.setdefault("logging", {})["json"] = True
    return overrides


async def _run(args: argparse.Namespace) -> int:
    config = load_config(
        profile=args.profile, extra_file=args.config, overrides=build_overrides(args)
    )
    setup_logging(
        level=str(config.get("logging.level", "INFO")),
        json_logs=bool(config.get("logging.json", False)),
    )

    mode_texte = args.text or bool(args.once)
    assistant = build_assistant(
        config,
        voice=not mode_texte,
        wav_in=args.wav_in,
        wav_out=args.wav_out,
        silent_output=args.muet,
    )
    try:
        if mode_texte:
            return await run_text_mode(assistant, once=args.once)
        return await run_voice_mode(
            assistant,
            once=bool(args.wav_in),
            push_to_talk=args.push_to_talk,
            use_wake=not args.no_wake,
        )
    finally:
        await assistant.aclose()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=args.log_level or "INFO", json_logs=args.json_logs)
    if args.devices:
        print(list_devices())
        return 0
    try:
        return asyncio.run(_run(args))
    except EngineUnavailable as exc:
        print(f"Étage audio indisponible : {exc}", file=sys.stderr)
        return 3
    except CapucineError as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Point d'entrée de Capucine.

    python main.py --text                 # boucle clavier, sans micro ni haut-parleur
    python main.py --text --llm mock      # sans aucun modèle de langage
    python main.py --text --once "quelle heure est-il"
    python main.py                        # mode vocal (étapes 2 et 3)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capucine.app import build_assistant, run_text_mode  # noqa: E402
from capucine.core.config import load_config  # noqa: E402
from capucine.core.errors import CapucineError  # noqa: E402
from capucine.core.logging import get_logger, setup_logging  # noqa: E402


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
    parser.add_argument("--plugins", metavar="DOSSIER", action="append",
                        help="dossier de plugins supplémentaire (répétable)")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--json-logs", action="store_true",
                        help="journal en JSON, une ligne par événement")
    return parser.parse_args(argv)


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.llm:
        overrides.setdefault("llm", {})["engine"] = args.llm
    if args.plugins:
        overrides.setdefault("plugins", {})["paths"] = list(args.plugins)
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
    logger = get_logger("main")

    assistant = build_assistant(config)
    try:
        if args.text or args.once:
            return await run_text_mode(assistant, once=args.once)
        logger.error(
            "Le mode vocal arrive à l'étape 2 (STT/TTS) et à l'étape 3 (mot d'éveil). "
            "Utilisez « python main.py --text » en attendant."
        )
        return 2
    finally:
        await assistant.aclose()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=args.log_level or "INFO", json_logs=args.json_logs)
    try:
        return asyncio.run(_run(args))
    except CapucineError as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Point d'entrée de l'application de bureau.

    python lily_bureau.py                 # la fenêtre
    python lily_bureau.py --profile pi    # avec un profil imposé
    python lily_bureau.py --llm mock      # sans modèle de langage

C'est le pendant de ``main.py`` : mêmes options de configuration, même
assistant, une fenêtre au lieu d'un terminal. Rien de ce qui suit ne touche au
cœur — l'interface consomme ce que la ligne de commande consomme déjà.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lily.core.config import load_config  # noqa: E402
from lily.core.errors import LilyError  # noqa: E402
from lily.core.logging import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lily-bureau",
        description="Lily — l'application de bureau.",
    )
    parser.add_argument("--profile", choices=["pc", "pi"],
                        help="profil de configuration (détecté automatiquement sinon)")
    parser.add_argument("--config", metavar="FICHIER",
                        help="fichier TOML supplémentaire, appliqué par-dessus le profil")
    parser.add_argument("--llm", metavar="MOTEUR",
                        help="remplace llm.engine : ollama, llamacpp ou mock")
    parser.add_argument("--atelier", metavar="DOSSIER", action="append",
                        help="dossier de travail (répétable). À défaut, Documents/Lily.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    surcharges: dict = {}
    if args.llm:
        surcharges.setdefault("llm", {})["engine"] = args.llm
    if args.atelier:
        surcharges.setdefault("atelier", {})["racines"] = list(args.atelier)

    try:
        config = load_config(
            profile=args.profile, extra_file=args.config, overrides=surcharges
        )
    except LilyError as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
        return 2

    try:
        from lily.bureau.fenetre import lancer
    except ImportError as exc:
        print(
            "L'interface graphique réclame PySide6.\n"
            '  pip install -e ".[bureau]"\n'
            f"  ({exc})",
            file=sys.stderr,
        )
        return 3
    return lancer(config)


if __name__ == "__main__":
    raise SystemExit(main())

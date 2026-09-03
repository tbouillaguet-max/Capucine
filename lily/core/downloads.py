"""Récupération des poids : voix Piper, modèles Whisper, modèles Vosk.

Rien n'est téléchargé automatiquement au démarrage. C'est une commande
explicite, parce qu'un assistant censé fonctionner le Wi-Fi coupé ne doit
jamais sortir sur le réseau sans qu'on le lui demande.

    python -m lily.core.downloads voix fr_FR-siwis-medium
    python -m lily.core.downloads whisper small
    python -m lily.core.downloads vosk
    python -m lily.core.downloads tout --profile pi
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import PROJECT_ROOT, load_config
from .logging import get_logger, setup_logging

logger = get_logger("telechargement")

DOSSIER_MODELES = PROJECT_ROOT / "models"
VOSK_MODELES = {
    "small-fr": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "fr": "https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip",
}


def telecharger_voix(nom: str, dossier: Path | None = None) -> Path:
    """Télécharge une voix Piper (fichiers .onnx et .onnx.json)."""
    dossier = dossier or DOSSIER_MODELES / "piper"
    dossier.mkdir(parents=True, exist_ok=True)
    cible = dossier / f"{nom}.onnx"
    if cible.exists():
        logger.info("Voix déjà présente : %s", cible)
        return cible
    try:
        from piper.download_voices import download_voice
    except ImportError as exc:
        raise SystemExit(
            "Le paquet « piper-tts » est absent. Installez-le avec : pip install piper-tts"
        ) from exc
    logger.info("Téléchargement de la voix %s vers %s", nom, dossier)
    download_voice(nom, dossier)
    return cible


def telecharger_whisper(taille: str, dossier: Path | None = None) -> Path:
    """Pré-charge un modèle faster-whisper dans le dossier du projet."""
    dossier = dossier or DOSSIER_MODELES / "whisper"
    dossier.mkdir(parents=True, exist_ok=True)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "Le paquet « faster-whisper » est absent. "
            "Installez-le avec : pip install faster-whisper"
        ) from exc
    logger.info("Téléchargement du modèle Whisper « %s » vers %s", taille, dossier)
    # L'instanciation déclenche le téléchargement puis garde le modèle en cache.
    WhisperModel(taille, device="cpu", compute_type="int8", download_root=str(dossier))
    return dossier


def telecharger_vosk(variante: str = "small-fr", dossier: Path | None = None) -> Path:
    """Télécharge et décompresse un modèle Vosk français."""
    if variante not in VOSK_MODELES:
        raise SystemExit(f"Variante Vosk inconnue : {variante}. Choix : {', '.join(VOSK_MODELES)}")
    dossier = dossier or DOSSIER_MODELES / "vosk"
    dossier.mkdir(parents=True, exist_ok=True)
    url = VOSK_MODELES[variante]
    archive = dossier / Path(url).name
    cible = dossier / archive.stem
    if cible.is_dir():
        logger.info("Modèle Vosk déjà présent : %s", cible)
        return cible

    logger.info("Téléchargement de %s", url)
    with urllib.request.urlopen(url) as reponse, archive.open("wb") as sortie:  # noqa: S310
        shutil.copyfileobj(reponse, sortie)
    logger.info("Décompression vers %s", dossier)
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(dossier)
    archive.unlink(missing_ok=True)
    return cible


def tout_telecharger(profil: str | None = None) -> None:
    """Récupère ce que la configuration active réclame réellement."""
    config = load_config(profile=profil)
    voix = str(config.get("tts.voice", "fr_FR-siwis-medium"))
    telecharger_voix(voix, config.resolve_path("tts.models_dir"))

    moteur_stt = str(config.get("stt.engine", "faster-whisper"))
    if moteur_stt == "vosk":
        telecharger_vosk()
    else:
        telecharger_whisper(str(config.get("stt.model", "small")), config.resolve_path("stt.models_dir"))
    logger.info("Terminé. Profil « %s ».", config.get("profile"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lily.core.downloads",
        description="Télécharge les poids dont Lily a besoin. Rien n'est fait automatiquement.",
    )
    sous = parser.add_subparsers(dest="commande", required=True)

    p_voix = sous.add_parser("voix", help="voix Piper")
    p_voix.add_argument("nom", nargs="?", default="fr_FR-siwis-medium")

    p_whisper = sous.add_parser("whisper", help="modèle faster-whisper")
    p_whisper.add_argument("taille", nargs="?", default="small",
                           choices=["tiny", "base", "small", "medium", "large-v3"])

    p_vosk = sous.add_parser("vosk", help="modèle Vosk français")
    p_vosk.add_argument("variante", nargs="?", default="small-fr", choices=list(VOSK_MODELES))

    p_tout = sous.add_parser("tout", help="tout ce que la configuration active réclame")
    p_tout.add_argument("--profile", choices=["pc", "pi"])

    args = parser.parse_args(argv)
    setup_logging("INFO")

    if args.commande == "voix":
        print(f"Voix prête : {telecharger_voix(args.nom)}")
    elif args.commande == "whisper":
        print(f"Modèle Whisper prêt dans : {telecharger_whisper(args.taille)}")
    elif args.commande == "vosk":
        print(f"Modèle Vosk prêt : {telecharger_vosk(args.variante)}")
    else:
        tout_telecharger(args.profile)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

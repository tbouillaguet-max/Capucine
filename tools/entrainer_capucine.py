#!/usr/bin/env python3
"""Entraînement du modèle de mot d'éveil « Capucine » pour openWakeWord.

    python tools/entrainer_capucine.py preparer      # config + état des prérequis
    python tools/entrainer_capucine.py echantillons  # positifs synthétiques, en français
    python tools/entrainer_capucine.py entrainer     # lance l'entraînement officiel
    python tools/entrainer_capucine.py installer     # copie le modèle dans models/wake/
    python tools/entrainer_capucine.py essayer a.wav # vérifie sur un enregistrement

Une mise en garde d'emblée, pour éviter la mauvaise surprise : le chemin
officiel d'openWakeWord repose sur une génération de données synthétiques
pensée pour l'anglais (``piper-sample-generator`` et son point de contrôle
LibriTTS). Pour un mot français, la qualité des positifs est le facteur
limitant. La sous-commande ``echantillons`` produit donc des positifs avec les
**voix françaises de Piper** déjà installées, en variant débit, hauteur et
niveau — c'est ce qui manque le plus au pipeline d'origine.

Tant que le modèle n'est pas prêt, Capucine bascule automatiquement sur le
repli Vosk à grammaire restreinte. Ce n'est pas une panne : c'est un état
normal du projet, et il est parfaitement utilisable.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import wave
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

DOSSIER_WAKE = RACINE / "models" / "wake"
DOSSIER_TRAVAIL = DOSSIER_WAKE / "entrainement"
CONFIG = DOSSIER_TRAVAIL / "capucine.yaml"
MOT = "capucine"

# Voix françaises de Piper : plus il y en a, plus le modèle généralise.
VOIX_FR = [
    "fr_FR-siwis-medium",
    "fr_FR-upmc-medium",
    "fr_FR-gilles-low",
    "fr_FR-tom-medium",
]

# Prononciations et variantes à couvrir côté positifs.
PHRASES_POSITIVES = [
    "Capucine",
    "Capucine,",
    "Capucine ?",
    "Hé Capucine",
    "Dis Capucine",
    "Ma Capucine",
]

# Ce que le modèle doit apprendre à NE PAS déclencher. En français, les
# confusions viennent surtout de « capuc- » et de « -ine ».
NEGATIFS = [
    "capucin", "capuche", "capucines", "cabine", "câpres", "capote",
    "cousine", "cuisine", "copine", "colline", "combine", "machine",
    "la capuche du manteau", "un capucin en robe de bure",
    "je suis dans la cuisine", "ferme la cabine", "ma cousine arrive",
]


def _yaml(donnees: dict) -> str:
    """Sérialiseur minimal : évite d'imposer PyYAML pour écrire un fichier."""
    lignes: list[str] = []
    for cle, valeur in donnees.items():
        if isinstance(valeur, list):
            if not valeur:
                lignes.append(f"{cle}: []")
                continue
            lignes.append(f"{cle}:")
            lignes.extend(f"  - {json.dumps(v, ensure_ascii=False)}" for v in valeur)
        elif isinstance(valeur, str):
            lignes.append(f"{cle}: {json.dumps(valeur, ensure_ascii=False)}")
        elif isinstance(valeur, bool):
            lignes.append(f"{cle}: {str(valeur).lower()}")
        else:
            lignes.append(f"{cle}: {valeur}")
    return "\n".join(lignes) + "\n"


def commande_preparer(args: argparse.Namespace) -> int:
    DOSSIER_TRAVAIL.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": MOT,
        "target_phrase": [MOT],
        "custom_negative_phrases": NEGATIFS,
        "model_type": "dnn",
        "layer_size": 32,
        "total_length": 16000 * 3,          # 3 s de contexte, comme les modèles livrés
        "n_samples": args.positifs,
        "n_samples_val": max(200, args.positifs // 20),
        "steps": args.steps,
        "batch_n_per_class": {"adversarial_negative": 50, "positive": 50},
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
        "augmentation_rounds": 1,
        "augmentation_batch_size": 16,
        "tts_batch_size": 50,
        "output_dir": str(DOSSIER_TRAVAIL),
        "piper_sample_generator_path": str(args.generateur or DOSSIER_TRAVAIL / "piper-sample-generator"),
        "rir_paths": [str(DOSSIER_TRAVAIL / "rir")],
        "background_paths": [str(DOSSIER_TRAVAIL / "bruit")],
        "background_paths_duplication_rate": [1],
        "false_positive_validation_data_path": str(DOSSIER_TRAVAIL / "validation_negatifs.npy"),
        "feature_data_files": {},
    }
    CONFIG.write_text(_yaml(config), encoding="utf-8")
    print(f"Configuration écrite : {CONFIG}\n")

    print("État des prérequis :")
    manquants: list[str] = []

    def verifier(nom: str, present: bool, remede: str) -> None:
        print(f"  {'✓' if present else '✗'} {nom}")
        if not present:
            manquants.append(f"  {nom}\n      {remede}")

    verifier("openwakeword", _module_present("openwakeword"),
             "pip install openwakeword")
    verifier("torch (entraînement seulement)", _module_present("torch"),
             "pip install torch --index-url https://download.pytorch.org/whl/cpu")
    verifier("piper-tts (génération des positifs)", _module_present("piper"),
             "pip install piper-tts")
    verifier("générateur d'échantillons Piper",
             Path(config["piper_sample_generator_path"]).is_dir(),
             "git clone https://github.com/rhasspy/piper-sample-generator "
             f"{config['piper_sample_generator_path']}")
    verifier("réponses impulsionnelles (rir/)", _contient_des_wav(DOSSIER_TRAVAIL / "rir"),
             "déposez des WAV de réverbération (MIT IR Survey, openSLR 28) dans "
             f"{DOSSIER_TRAVAIL / 'rir'}")
    verifier("bruits de fond (bruit/)", _contient_des_wav(DOSSIER_TRAVAIL / "bruit"),
             "déposez des WAV d'ambiance (FSD50K, DEMAND) dans "
             f"{DOSSIER_TRAVAIL / 'bruit'}")

    if manquants:
        print("\nÀ compléter avant d'entraîner :")
        print("\n".join(manquants))
        print(
            "\nLes réverbérations et les bruits ne sont pas facultatifs : sans eux, "
            "le modèle apprend une pièce et un micro, pas un mot."
        )
    else:
        print("\nTout est prêt : python tools/entrainer_capucine.py entrainer")
    print(
        "\nEn attendant, Capucine utilise le repli Vosk (wake.engine = \"vosk\").\n"
        "Vous pouvez déjà produire les positifs français :\n"
        "  python tools/entrainer_capucine.py echantillons"
    )
    return 0


def commande_echantillons(args: argparse.Namespace) -> int:
    """Positifs synthétiques avec les voix françaises de Piper.

    C'est le maillon faible du pipeline d'origine pour un mot français : le
    générateur officiel parle anglais. On varie le débit, la couleur et le
    niveau pour que le modèle n'apprenne pas une seule diction.
    """
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError:
        print("Le paquet « piper-tts » est absent. Installez-le avec : pip install piper-tts",
              file=sys.stderr)
        return 2

    dossier_voix = RACINE / "models" / "piper"
    disponibles = [nom for nom in VOIX_FR if (dossier_voix / f"{nom}.onnx").exists()]
    if not disponibles:
        print(
            "Aucune voix française trouvée dans models/piper. Téléchargez-en :\n"
            + "\n".join(f"  python -m capucine.core.downloads voix {nom}" for nom in VOIX_FR),
            file=sys.stderr,
        )
        return 2

    sortie = Path(args.sortie or DOSSIER_TRAVAIL / "positifs")
    sortie.mkdir(parents=True, exist_ok=True)
    alea = random.Random(args.graine)
    produits = 0

    print(f"{len(disponibles)} voix : {', '.join(disponibles)}")
    for nom in disponibles:
        voix = PiperVoice.load(dossier_voix / f"{nom}.onnx")
        for index in range(args.par_voix):
            phrase = alea.choice(PHRASES_POSITIVES)
            config = SynthesisConfig(
                # Un mot d'éveil est prononcé vite, parfois traîné : il faut
                # les deux dans le corpus.
                length_scale=alea.uniform(0.75, 1.35),
                noise_scale=alea.uniform(0.5, 0.9),
                noise_w_scale=alea.uniform(0.6, 1.0),
                volume=alea.uniform(0.5, 1.0),
            )
            morceaux = list(voix.synthesize(phrase, syn_config=config))
            if not morceaux:
                continue
            chemin = sortie / f"{nom}_{index:04d}.wav"
            with wave.open(str(chemin), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(morceaux[0].sample_rate)
                for morceau in morceaux:
                    handle.writeframes(morceau.audio_int16_bytes)
            produits += 1

    print(f"{produits} échantillons positifs écrits dans {sortie}")
    print(
        "Passez-les ensuite par l'augmentation d'openWakeWord (réverbération et "
        "bruit de fond) : sans elle, le modèle ne tiendra que dans la pièce où "
        "il a été enregistré."
    )
    return 0


def commande_entrainer(args: argparse.Namespace) -> int:
    if not CONFIG.exists():
        print("Configuration absente. Lancez d'abord : "
              "python tools/entrainer_capucine.py preparer", file=sys.stderr)
        return 2
    if not _module_present("torch"):
        print("L'entraînement réclame torch. "
              "pip install torch --index-url https://download.pytorch.org/whl/cpu",
              file=sys.stderr)
        return 2

    commande = [
        sys.executable, "-m", "openwakeword.train",
        "--training_config", str(CONFIG),
    ]
    if not args.reprendre:
        commande += ["--generate_clips", "--augment_clips"]
    commande.append("--train_model")

    print("Lancement :", " ".join(commande))
    print("Comptez plusieurs heures sur CPU. Le modèle sort dans", DOSSIER_TRAVAIL / MOT)
    return subprocess.call(commande)


def commande_installer(_args: argparse.Namespace) -> int:
    candidats = sorted(DOSSIER_TRAVAIL.rglob(f"{MOT}.onnx")) + \
        sorted(DOSSIER_TRAVAIL.rglob(f"{MOT}.tflite"))
    if not candidats:
        print(f"Aucun modèle entraîné trouvé sous {DOSSIER_TRAVAIL}", file=sys.stderr)
        return 1
    source = candidats[0]
    DOSSIER_WAKE.mkdir(parents=True, exist_ok=True)
    cible = DOSSIER_WAKE / source.name
    shutil.copy2(source, cible)
    print(f"Modèle installé : {cible}")
    print("Capucine l'utilisera au prochain démarrage (wake.engine = \"openwakeword\").")
    return 0


def commande_essayer(args: argparse.Namespace) -> int:
    """Passe un enregistrement dans le modèle et affiche le score maximum."""
    from capucine.core.audio import Rechunker, WavFileInput, record
    from capucine.core.engines.wake.openwakeword import OpenWakeWordEngine

    moteur = OpenWakeWordEngine(models_dir=DOSSIER_WAKE, threshold=args.seuil, debounce_s=0.0)
    if not moteur.available():
        print(f"Aucun modèle « {MOT} » dans {DOSSIER_WAKE}.", file=sys.stderr)
        return 2

    source = WavFileInput(args.fichier)
    source.start()
    audio = record(source, max_seconds=600)
    rechunker = Rechunker(moteur.frame_size)
    detections = 0
    for trame in rechunker.push(audio.pcm):
        if moteur.process(trame) is not None:
            detections += 1
    print(f"{args.fichier} : {detections} détection(s) au seuil {args.seuil}")
    return 0 if detections else 1


def _module_present(nom: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(nom) is not None
    except (ImportError, ValueError):
        return False


def _contient_des_wav(dossier: Path) -> bool:
    return dossier.is_dir() and any(dossier.rglob("*.wav"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="entrainer_capucine",
        description="Entraîne le modèle de mot d'éveil « Capucine » pour openWakeWord.",
    )
    sous = parser.add_subparsers(dest="commande", required=True)

    p = sous.add_parser("preparer", help="écrit la configuration et vérifie les prérequis")
    p.add_argument("--positifs", type=int, default=5000)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--generateur", type=Path, help="chemin de piper-sample-generator")
    p.set_defaults(fonction=commande_preparer)

    p = sous.add_parser("echantillons", help="positifs synthétiques avec les voix françaises")
    p.add_argument("--par-voix", type=int, default=250)
    p.add_argument("--sortie", type=Path)
    p.add_argument("--graine", type=int, default=0)
    p.set_defaults(fonction=commande_echantillons)

    p = sous.add_parser("entrainer", help="lance l'entraînement officiel openWakeWord")
    p.add_argument("--reprendre", action="store_true",
                   help="réutilise les clips déjà générés et augmentés")
    p.set_defaults(fonction=commande_entrainer)

    p = sous.add_parser("installer", help="copie le modèle entraîné dans models/wake/")
    p.set_defaults(fonction=commande_installer)

    p = sous.add_parser("essayer", help="teste le modèle sur un enregistrement")
    p.add_argument("fichier", type=Path)
    p.add_argument("--seuil", type=float, default=0.5)
    p.set_defaults(fonction=commande_essayer)

    args = parser.parse_args(argv)
    return int(args.fonction(args))


if __name__ == "__main__":
    raise SystemExit(main())

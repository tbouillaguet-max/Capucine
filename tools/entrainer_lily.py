#!/usr/bin/env python3
"""Entraînement du modèle de mot d'éveil « Lily » pour openWakeWord.

    python tools/entrainer_lily.py preparer      # config + état des prérequis
    python tools/entrainer_lily.py echantillons  # positifs synthétiques, en français
    python tools/entrainer_lily.py entrainer     # lance l'entraînement officiel
    python tools/entrainer_lily.py installer     # copie le modèle dans models/wake/
    python tools/entrainer_lily.py essayer a.wav # vérifie sur un enregistrement
    python tools/entrainer_lily.py corpus        # verse ce qu'elle a entendu chez vous
    python tools/entrainer_lily.py seuil         # mesure VOTRE seuil, au lieu de le deviner

Une mise en garde d'emblée, pour éviter la mauvaise surprise : le chemin
officiel d'openWakeWord repose sur une génération de données synthétiques
pensée pour l'anglais (``piper-sample-generator`` et son point de contrôle
LibriTTS). Pour un mot français, la qualité des positifs est le facteur
limitant. La sous-commande ``echantillons`` produit donc des positifs avec les
**voix françaises de Piper** déjà installées, en variant débit, hauteur et
niveau — c'est ce qui manque le plus au pipeline d'origine.

Tant que le modèle n'est pas prêt, Lily bascule automatiquement sur le
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
CONFIG = DOSSIER_TRAVAIL / "lily.yaml"
MOT = "lily"
# Là où Lily dépose ce qu'elle a entendu en usage réel, quand [corpus]
# actif = true. Ce sont les seuls exemples de VOTRE voix et de VOS faux
# déclenchements — ceux qu'aucune synthèse ne sait produire.
CORPUS_PAR_DEFAUT = Path.home() / ".lily" / "corpus"

# Voix françaises de Piper : plus il y en a, plus le modèle généralise.
VOIX_FR = [
    "fr_FR-siwis-medium",
    "fr_FR-upmc-medium",
    "fr_FR-gilles-low",
    "fr_FR-tom-medium",
]

# Prononciations et variantes à couvrir côté positifs.
PHRASES_POSITIVES = [
    "Lily",
    "Lily,",
    "Lily ?",
    "Hé Lily",
    "Dis Lily",
    "Ma Lily",
]

# Ce que le modèle doit apprendre à NE PAS déclencher. « Lily » est un nom
# court, donc très exposé : en français les confusions viennent de « li- »,
# de « -lie » et de la liaison « il y a ».
NEGATIFS = [
    "lilas", "limite", "ligne", "litige", "délit", "joli", "poli",
    "Émilie", "Nathalie", "Julie", "Mélissa", "la vie", "l'île",
    "j'ai vu des lilas", "il y a du monde", "c'est joli",
    "Émilie arrive", "c'est la limite", "elle lit un livre",
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
        print("\nTout est prêt : python tools/entrainer_lily.py entrainer")
    print(
        "\nEn attendant, Lily utilise le repli Vosk (wake.engine = \"vosk\").\n"
        "Vous pouvez déjà produire les positifs français :\n"
        "  python tools/entrainer_lily.py echantillons"
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
            + "\n".join(f"  python -m lily.core.downloads voix {nom}" for nom in VOIX_FR),
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
              "python tools/entrainer_lily.py preparer", file=sys.stderr)
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
    print("Lily l'utilisera au prochain démarrage (wake.engine = \"openwakeword\").")
    return 0


def commande_essayer(args: argparse.Namespace) -> int:
    """Passe un enregistrement dans le modèle et affiche le score maximum."""
    from lily.core.audio import Rechunker, WavFileInput, record
    from lily.core.engines.wake.openwakeword import OpenWakeWordEngine

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


def commande_corpus(args: argparse.Namespace) -> int:
    """Verse les extraits collectés en usage réel dans le jeu d'entraînement.

    Deux choses, et une seule est automatique — le dire franchement évite la
    mauvaise surprise :

    * les **vrais éveils** rejoignent les positifs, où le pipeline
      d'openWakeWord les augmente comme les autres : c'est fait ici ;
    * les **faux positifs** sont des négatifs difficiles, et openWakeWord les
      attend sous forme de caractéristiques précalculées (un ``.npy``), pas de
      WAV. Ils sont rangés à part, prêts pour cette étape, qui reste à faire
      dans le carnet officiel.
    """
    corpus = Path(args.corpus or CORPUS_PAR_DEFAUT).expanduser()
    if not corpus.is_dir():
        print(f"Aucun corpus dans {corpus}.", file=sys.stderr)
        print("Activez-le d'abord : [corpus] actif = true dans la configuration.",
              file=sys.stderr)
        return 2

    eveils = sorted((corpus / "eveils").glob("*.wav"))
    faux = sorted((corpus / "faux_positifs").glob("*.wav"))
    if not eveils and not faux:
        print(f"Corpus vide : {corpus}")
        print("Parlez-lui quelques jours, elle le remplira toute seule.")
        return 1

    positifs = Path(args.sortie or DOSSIER_TRAVAIL / "positifs")
    negatifs = DOSSIER_TRAVAIL / "negatifs_reels"
    positifs.mkdir(parents=True, exist_ok=True)
    negatifs.mkdir(parents=True, exist_ok=True)

    for source in eveils:
        shutil.copy2(source, positifs / f"reel_{source.name}")
    for source in faux:
        shutil.copy2(source, negatifs / source.name)

    print(f"{len(eveils)} vrai(s) éveil(s) → {positifs}")
    print(f"{len(faux)} faux positif(s) → {negatifs}")
    if eveils:
        print(
            "\nLes positifs réels valent plusieurs centaines de positifs "
            "synthétiques : ils portent votre timbre, votre débit et votre pièce."
        )
    if faux:
        print(
            f"\nLes {len(faux)} faux positifs sont le vrai trésor : un modèle de mot "
            "d'éveil échoue presque toujours par excès de déclenchements, jamais\n"
            "par manque de positifs. openWakeWord les veut en caractéristiques "
            "précalculées — passez-les par openwakeword.data.compute_features\n"
            "et pointez false_positive_validation_data_path dessus dans "
            f"{CONFIG.name}."
        )
    print("\nEnsuite : python tools/entrainer_lily.py entrainer")
    return 0


def commande_seuil(args: argparse.Namespace) -> int:
    """Mesure le seuil d'éveil sur le corpus étiqueté, au lieu de le deviner.

    Le seuil livré (0,5) est un compromis pour une voix moyenne dans une pièce
    moyenne. Vous n'avez ni l'une ni l'autre. Cette commande passe le modèle
    courant sur vos propres enregistrements et dit à partir de quelle valeur
    il vous reconnaît sans se réveiller pour rien.
    """
    from lily.core.audio import Rechunker, WavFileInput, record
    from lily.core.engines.wake.openwakeword import OpenWakeWordEngine

    corpus = Path(args.corpus or CORPUS_PAR_DEFAUT).expanduser()
    eveils = sorted((corpus / "eveils").glob("*.wav"))
    faux = sorted((corpus / "faux_positifs").glob("*.wav"))
    if not eveils:
        print(f"Aucun vrai éveil dans {corpus / 'eveils'} : rien à mesurer.",
              file=sys.stderr)
        return 2

    moteur = OpenWakeWordEngine(models_dir=DOSSIER_WAKE, threshold=1.1, debounce_s=0.0)
    if not moteur.available():
        print(f"Aucun modèle « {MOT} » dans {DOSSIER_WAKE}.", file=sys.stderr)
        return 2

    def scores_de(fichiers: list[Path]) -> list[float]:
        resultats: list[float] = []
        for fichier in fichiers:
            source = WavFileInput(fichier)
            source.start()
            audio = record(source, max_seconds=60)
            moteur.reset()
            rechunker = Rechunker(moteur.frame_size)
            maximum = 0.0
            for trame in rechunker.push(audio.pcm):
                maximum = max(maximum, moteur.score(trame))
            resultats.append(maximum)
        return resultats

    positifs = scores_de(eveils)
    negatifs = scores_de(faux)
    print(f"{len(positifs)} vrais éveils, {len(negatifs)} faux positifs\n")
    print(f"  éveils        : min {min(positifs):.2f}  médian "
          f"{sorted(positifs)[len(positifs) // 2]:.2f}  max {max(positifs):.2f}")
    if negatifs:
        print(f"  faux positifs : min {min(negatifs):.2f}  médian "
              f"{sorted(negatifs)[len(negatifs) // 2]:.2f}  max {max(negatifs):.2f}")

    print("\n  seuil   détectés   faux déclenchements")
    meilleur, meilleur_score = None, -1.0
    for centieme in range(20, 96, 5):
        seuil = centieme / 100
        vrais = sum(1 for valeur in positifs if valeur >= seuil)
        rates = sum(1 for valeur in negatifs if valeur >= seuil)
        print(f"  {seuil:.2f}    {vrais:3d}/{len(positifs):<3d}    {rates:3d}/{len(negatifs) or 0:<3d}")
        # Un éveil manqué se rattrape en répétant ; un faux déclenchement
        # coupe la parole et fait peur. On le pénalise trois fois plus.
        qualite = vrais / len(positifs) - 3.0 * (rates / len(negatifs) if negatifs else 0.0)
        if qualite > meilleur_score:
            meilleur, meilleur_score = seuil, qualite

    print(f"\nSeuil conseillé sur VOTRE corpus : {meilleur:.2f}")
    print(f"À reporter dans la configuration :\n\n  [wake]\n  threshold = {meilleur:.2f}")
    if len(positifs) < 20:
        print(
            f"\n⚠ {len(positifs)} exemples seulement : le conseil est indicatif. "
            "Reprenez cette mesure après quelques jours d'usage."
        )
    return 0


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
        prog="entrainer_lily",
        description="Entraîne le modèle de mot d'éveil « Lily » pour openWakeWord.",
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

    p = sous.add_parser("corpus", help="verse les extraits collectés en usage réel dans le jeu")
    p.add_argument("--corpus", type=Path, help=f"défaut : {CORPUS_PAR_DEFAUT}")
    p.add_argument("--sortie", type=Path, help="dossier des positifs")
    p.set_defaults(fonction=commande_corpus)

    p = sous.add_parser("seuil", help="mesure le seuil d'éveil sur le corpus étiqueté")
    p.add_argument("--corpus", type=Path, help=f"défaut : {CORPUS_PAR_DEFAUT}")
    p.set_defaults(fonction=commande_seuil)

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

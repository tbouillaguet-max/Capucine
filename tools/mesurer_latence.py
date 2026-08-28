#!/usr/bin/env python3
"""Banc de mesure : où part le temps, étage par étage.

    python tools/mesurer_latence.py                  # tout ce qui est disponible
    python tools/mesurer_latence.py --profile pi
    python tools/mesurer_latence.py --wav essai.wav  # transcrire un vrai enregistrement
    python tools/mesurer_latence.py --json           # pour comparer deux machines

Deux familles de chiffres, et elles ne se lisent pas de la même façon.

**Les étages permanents** — mot d'éveil et VAD — tournent en continu. Ce qui
compte pour eux n'est pas la latence mais le **facteur temps réel** : le
rapport entre le temps de calcul et la durée d'audio traitée. À 0,10, un cœur
sur dix est occupé en permanence ; au-dessus de 0,5, le Pi n'aura plus de
souffle pour transcrire.

**Les étages à la demande** — transcription, routage, plugin, synthèse — ne
coûtent que pendant un tour. Là, c'est la latence qui compte, et surtout le
**temps avant la première parole** : c'est lui que l'utilisateur ressent.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from capucine.core.audio import AudioBuffer, WavFileInput, record  # noqa: E402
from capucine.core.config import load_config  # noqa: E402
from capucine.core.engines.factory import (  # noqa: E402
    build_llm,
    build_stt,
    build_tts,
    build_vad,
    build_wake,
)
from capucine.core.errors import CapucineError  # noqa: E402
from capucine.core.logging import setup_logging  # noqa: E402
from capucine.core.machine import conseils, decrire  # noqa: E402
from capucine.core.registry import PluginRegistry  # noqa: E402
from capucine.core.router import Router  # noqa: E402

PHRASE = "Il est neuf heures vingt. Le minuteur des pâtes sonnera dans trois minutes."


@dataclass
class Mesure:
    etage: str
    unite: str                       # « latence » ou « temps réel »
    p50_ms: float | None = None
    p90_ms: float | None = None
    facteur_temps_reel: float | None = None
    note: str = ""
    disponible: bool = True
    details: dict = field(default_factory=dict)


def _chronometrer(fonction, repetitions: int) -> list[float]:
    durees = []
    for _ in range(repetitions):
        depart = time.perf_counter()
        fonction()
        durees.append((time.perf_counter() - depart) * 1000.0)
    return durees


def _stats(durees: list[float]) -> tuple[float, float]:
    ordonnees = sorted(durees)
    p50 = statistics.median(ordonnees)
    rang = min(len(ordonnees) - 1, int(round(0.9 * (len(ordonnees) - 1))))
    return round(p50, 2), round(ordonnees[rang], 2)


def _indisponible(etage: str, unite: str, raison: str) -> Mesure:
    return Mesure(etage=etage, unite=unite, note=raison, disponible=False)


# --- étages permanents ------------------------------------------------------

def mesurer_eveil(config, repetitions: int) -> Mesure:
    moteur = build_wake(config)
    if moteur is None:
        return _indisponible(
            "mot d'éveil", "temps réel",
            "aucun moteur (modèle non entraîné et Vosk absent)",
        )
    trame = b"\x00\x00" * moteur.frame_size
    moteur.process(trame)  # préchauffage : le premier appel charge le modèle
    durees = _chronometrer(lambda: moteur.process(trame), repetitions)
    p50, p90 = _stats(durees)
    duree_trame_ms = moteur.frame_size / moteur.sample_rate * 1000
    moteur.close()
    return Mesure(
        etage="mot d'éveil", unite="temps réel", p50_ms=p50, p90_ms=p90,
        facteur_temps_reel=round(p50 / duree_trame_ms, 4),
        details={"moteur": moteur.describe(), "trame_ms": round(duree_trame_ms, 1)},
    )


def mesurer_vad(config, repetitions: int) -> Mesure:
    moteur = build_vad(config)
    trame = b"\x00\x00" * moteur.frame_size
    moteur.speech_probability(trame)
    durees = _chronometrer(lambda: moteur.speech_probability(trame), repetitions)
    p50, p90 = _stats(durees)
    duree_trame_ms = moteur.frame_size / moteur.sample_rate * 1000
    moteur.close()
    return Mesure(
        etage="VAD", unite="temps réel", p50_ms=p50, p90_ms=p90,
        facteur_temps_reel=round(p50 / duree_trame_ms, 4),
        details={"moteur": moteur.describe(), "trame_ms": round(duree_trame_ms, 1)},
    )


# --- étages à la demande ----------------------------------------------------

def _audio_de_test(chemin: Path | None, tts) -> AudioBuffer | None:
    """De quoi transcrire : un enregistrement fourni, ou la voix de Capucine.

    Faire dire la phrase par Piper puis la faire transcrire par Whisper donne
    un aller-retour honnête et reproductible, sans micro.
    """
    if chemin is not None:
        source = WavFileInput(chemin)
        source.start()
        return record(source, max_seconds=60)
    if tts is None:
        return None
    morceaux = list(tts.synthesize(PHRASE))
    if not morceaux:
        return None
    return AudioBuffer(b"".join(m.pcm for m in morceaux), morceaux[0].sample_rate)


def mesurer_transcription(config, audio: AudioBuffer | None, repetitions: int) -> Mesure:
    if audio is None:
        return _indisponible("transcription", "latence", "aucun audio de test")
    try:
        moteur = build_stt(config)
    except CapucineError as exc:
        return _indisponible("transcription", "latence", str(exc)[:120])

    moteur.warmup()
    resultat = moteur.transcribe(audio)
    durees = _chronometrer(lambda: moteur.transcribe(audio), max(1, repetitions))
    p50, p90 = _stats(durees)
    moteur.close()
    return Mesure(
        etage="transcription", unite="latence", p50_ms=p50, p90_ms=p90,
        facteur_temps_reel=round(p50 / (audio.duration_s * 1000), 3) if audio.duration_s else None,
        details={
            "moteur": moteur.describe(),
            "audio_s": round(audio.duration_s, 2),
            "texte": resultat.text[:80],
        },
    )


def mesurer_routage(config, registry: PluginRegistry, repetitions: int) -> list[Mesure]:
    llm = build_llm(config)
    options = config.section("llm").get("router", {}) or {}
    routeur = Router(
        llm,
        direct_threshold=float(options.get("direct_threshold", 0.72)),
        shortlist_threshold=float(options.get("shortlist_threshold", 0.35)),
        shortlist_size=int(options.get("shortlist_size", 5)),
    )
    skills = registry.skills
    mesures: list[Mesure] = []

    # Étage déterministe : la phrase colle à un exemple, aucun modèle sollicité.
    durees = _chronometrer(lambda: routeur.route("quelle heure est-il", skills), repetitions)
    p50, p90 = _stats(durees)
    mesures.append(Mesure(
        etage="routage déterministe", unite="latence", p50_ms=p50, p90_ms=p90,
        details={"competences": len(skills)},
    ))

    # Étage LLM : deux générations contraintes.
    if llm.name == "mock":
        mesures.append(_indisponible(
            "routage par le modèle", "latence",
            "aucun modèle de langage joignable (repli factice)",
        ))
    else:
        phrase = "j'aimerais que tu me racontes ce que tu sais faire exactement"
        durees = _chronometrer(lambda: routeur.route(phrase, skills), max(1, repetitions // 4))
        p50, p90 = _stats(durees)
        mesures.append(Mesure(
            etage="routage par le modèle", unite="latence", p50_ms=p50, p90_ms=p90,
            details={"moteur": llm.describe()},
        ))
    llm.close()
    return mesures


def mesurer_plugins(registry: PluginRegistry, repetitions: int) -> Mesure:
    """Mesure les compétences sûres : sans argument obligatoire, sans effet
    irréversible. On ne veut pas effacer les notes de l'utilisateur pour
    connaître sa latence."""
    sures = [
        spec for spec in registry.skills.values()
        if not spec.required_parameters and not spec.confirm and not spec.isolate
    ]
    if not sures:
        return _indisponible("plugins", "latence", "aucune compétence sans argument")

    par_competence: dict[str, float] = {}
    toutes: list[float] = []
    for spec in sures:
        durees = _chronometrer(lambda s=spec: registry.call(s.name), max(1, repetitions // 20))
        p50, _ = _stats(durees)
        par_competence[spec.name] = p50
        toutes.extend(durees)
    p50, p90 = _stats(toutes)
    return Mesure(
        etage="exécution de plugin", unite="latence", p50_ms=p50, p90_ms=p90,
        details={"par_competence": {k: round(v, 2) for k, v in sorted(
            par_competence.items(), key=lambda paire: -paire[1])[:6]}},
    )


def mesurer_synthese(config, repetitions: int) -> list[Mesure]:
    tts = build_tts(config)
    if tts is None:
        return [_indisponible("synthèse, 1re phrase", "latence", "aucune voix installée")]

    list(tts.synthesize("Bonjour."))  # préchauffage

    def _premiere_phrase() -> None:
        for _ in tts.synthesize(PHRASE):
            return   # on s'arrête au premier morceau : c'est là qu'elle parle

    durees = _chronometrer(_premiere_phrase, max(1, repetitions // 10))
    p50, p90 = _stats(durees)
    premiere = Mesure(
        etage="synthèse, 1re phrase", unite="latence", p50_ms=p50, p90_ms=p90,
        details={"moteur": tts.describe()},
    )

    morceaux = list(tts.synthesize(PHRASE))
    duree_audio = sum(m.duration_s for m in morceaux)
    durees = _chronometrer(lambda: list(tts.synthesize(PHRASE)), max(1, repetitions // 10))
    p50_total, p90_total = _stats(durees)
    complete = Mesure(
        etage="synthèse, complète", unite="latence", p50_ms=p50_total, p90_ms=p90_total,
        facteur_temps_reel=round(p50_total / (duree_audio * 1000), 3) if duree_audio else None,
        details={"phrases": len(morceaux), "audio_s": round(duree_audio, 2)},
    )
    tts.close()
    return [premiere, complete]


# --- rapport ----------------------------------------------------------------

def _ligne(mesure: Mesure) -> str:
    if not mesure.disponible:
        return f"  {mesure.etage:<24} —          {mesure.note}"
    latence = f"{mesure.p50_ms:.0f} ms" if mesure.p50_ms >= 10 else f"{mesure.p50_ms:.1f} ms"
    if mesure.p50_ms >= 1000:
        latence = f"{mesure.p50_ms / 1000:.2f} s"
    facteur = ""
    if mesure.facteur_temps_reel is not None:
        facteur = f"  ×{mesure.facteur_temps_reel:.3f} temps réel"
    return f"  {mesure.etage:<24} {latence:>9}{facteur}"


def rapport(mesures: list[Mesure], machine, remarques: list[str], config) -> str:
    lignes = [
        "",
        f"Machine : {machine.resume()}",
        f"Profil  : {config.get('profile')}",
        "",
        "Étages permanents (tournent en continu ; le facteur temps réel compte)",
    ]
    lignes += [_ligne(m) for m in mesures if m.unite == "temps réel"]
    lignes += ["", "Étages à la demande (une fois par tour ; la latence compte)"]
    lignes += [_ligne(m) for m in mesures if m.unite == "latence"]

    permanents = sum(
        m.facteur_temps_reel or 0 for m in mesures
        if m.unite == "temps réel" and m.disponible
    )
    if permanents:
        lignes += [
            "",
            f"À vide, l'écoute occupe en permanence l'équivalent de "
            f"{permanents * 100:.1f} % d'un cœur.",
        ]

    etages_du_tour = ("transcription", "routage déterministe", "exécution de plugin",
                      "synthèse, 1re phrase")
    total = sum(
        m.p50_ms or 0 for m in mesures if m.disponible and m.etage in etages_du_tour
    )
    manquants = [
        m.etage for m in mesures if not m.disponible and m.etage in etages_du_tour
    ]
    if total and not manquants:
        lignes += [
            "",
            f"Tour typique (phrase entendue → première parole) : environ "
            f"{total / 1000:.2f} s.",
        ]
    elif manquants:
        lignes += [
            "",
            "Tour typique non calculable : " + ", ".join(manquants)
            + " n'a pas pu être mesuré sur cette machine.",
        ]

    if remarques:
        lignes += ["", "Remarques sur la configuration :"]
        lignes += [f"  • {remarque}" for remarque in remarques]
    lignes.append("")
    return "\n".join(lignes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mesurer_latence",
        description="Mesure la latence de chaque étage de Capucine sur cette machine.",
    )
    parser.add_argument("--profile", choices=["pc", "pi"])
    parser.add_argument("--config", metavar="FICHIER")
    parser.add_argument("--wav", type=Path, help="enregistrement à transcrire")
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="sortie lisible par une machine")
    parser.add_argument("--log-level", default="ERROR")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    config = load_config(profile=args.profile, extra_file=args.config)
    machine = decrire()

    registry = PluginRegistry(
        config.plugin_paths(), config=config,
        data_root=config.resolve_path("plugins.data_dir"),
    )
    registry.load_all()

    mesures: list[Mesure] = [
        mesurer_eveil(config, args.repetitions),
        mesurer_vad(config, args.repetitions),
    ]
    tts_pour_audio = build_tts(config)
    audio = _audio_de_test(args.wav, tts_pour_audio)
    if tts_pour_audio is not None:
        tts_pour_audio.close()

    mesures.append(mesurer_transcription(config, audio, max(1, args.repetitions // 20)))
    mesures += mesurer_routage(config, registry, args.repetitions)
    mesures.append(mesurer_plugins(registry, args.repetitions))
    mesures += mesurer_synthese(config, args.repetitions)

    remarques = conseils(config, machine)
    if args.json:
        print(json.dumps({
            "machine": {
                "resume": machine.resume(), "systeme": machine.systeme,
                "architecture": machine.architecture, "coeurs": machine.coeurs,
                "memoire_go": machine.memoire_go, "est_pi": machine.est_pi,
            },
            "profil": config.get("profile"),
            "mesures": [asdict(m) for m in mesures],
            "remarques": remarques,
        }, ensure_ascii=False, indent=2))
    else:
        print(rapport(mesures, machine, remarques, config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

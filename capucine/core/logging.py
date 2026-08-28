"""Journalisation structurée et chronométrage par étage du pipeline.

L'objectif est le profilage sur Raspberry Pi : à la fin de chaque tour, une
ligne unique donne la latence de chaque étage (éveil, transcription,
inférence, exécution du plugin, synthèse) et le total.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

LOGGER_NAME = "capucine"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _HumanFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG": "\033[2m", "INFO": "\033[0m", "WARNING": "\033[33m",
        "ERROR": "\033[31m", "CRITICAL": "\033[1;31m",
    }

    def __init__(self, color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extra = getattr(record, "extra_fields", None)
        if extra:
            line += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        if self.color:
            return f"{self._COLORS.get(record.levelname, '')}{line}\033[0m"
        return line


def setup_logging(level: str = "INFO", json_logs: bool = False, color: bool | None = None) -> None:
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        if color is None:
            color = sys.stderr.isatty()
        handler.setFormatter(_HumanFormatter(color))
    root.addHandler(handler)


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def log_with(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"extra_fields": fields})


@dataclass
class Statistiques:
    """Ce qu'on retient d'un étage : combien, et à quelle vitesse."""

    etage: str
    nombre: int
    p50_ms: float
    p90_ms: float
    max_ms: float


class LatencyBook:
    """Agrège les latences par étage, sur les N derniers tours.

    Une ligne de journal par tour suffit à comprendre un tour ; elle ne dit
    rien de la tendance. Sur Raspberry Pi, ce qu'on veut savoir c'est « où
    part la seconde et demie », et la réponse est une médiane, pas un cas.

    Les échantillons sont bornés : un assistant qui tourne des semaines ne
    doit pas grossir indéfiniment.
    """

    def __init__(self, max_samples: int = 200) -> None:
        self.max_samples = max_samples
        self._echantillons: dict[str, deque[float]] = {}
        self._verrou = threading.Lock()

    def record(self, etage: str, millisecondes: float) -> None:
        with self._verrou:
            file = self._echantillons.setdefault(etage, deque(maxlen=self.max_samples))
            file.append(float(millisecondes))

    def reset(self) -> None:
        with self._verrou:
            self._echantillons.clear()

    def snapshot(self) -> list[Statistiques]:
        with self._verrou:
            copie = {etage: list(valeurs) for etage, valeurs in self._echantillons.items()}
        stats = [
            Statistiques(
                etage=etage,
                nombre=len(valeurs),
                p50_ms=_centile(valeurs, 50),
                p90_ms=_centile(valeurs, 90),
                max_ms=max(valeurs),
            )
            for etage, valeurs in copie.items() if valeurs
        ]
        # Le plus coûteux d'abord : c'est ce qu'on veut lire en premier.
        stats.sort(key=lambda s: s.p50_ms, reverse=True)
        return stats

    def table(self) -> str:
        stats = self.snapshot()
        if not stats:
            return "Aucune latence mesurée pour l'instant."
        entete = f"{'étage':<18}{'n':>5}{'médiane':>12}{'p90':>12}{'max':>12}"
        lignes = [entete, "-" * len(entete)]
        for stat in stats:
            lignes.append(
                f"{stat.etage:<18}{stat.nombre:>5}"
                f"{_ms(stat.p50_ms):>12}{_ms(stat.p90_ms):>12}{_ms(stat.max_ms):>12}"
            )
        return "\n".join(lignes)


def _ms(millisecondes: float) -> str:
    """Une durée lisible : on ne lit pas « 1847 ms » aussi vite que « 1,85 s »."""
    if millisecondes >= 1000:
        return f"{millisecondes / 1000:.2f} s"
    if millisecondes >= 10:
        return f"{millisecondes:.0f} ms"
    return f"{millisecondes:.1f} ms"


def _centile(valeurs: list[float], centile: int) -> float:
    if not valeurs:
        return 0.0
    ordonnees = sorted(valeurs)
    rang = min(len(ordonnees) - 1, int(round(centile / 100 * (len(ordonnees) - 1))))
    return round(ordonnees[rang], 1)


_LIVRE = LatencyBook()


def get_latency_book() -> LatencyBook:
    """Le carnet de latences partagé par tout le processus."""
    return _LIVRE


@dataclass
class TurnTelemetry:
    """Chronomètre les étages d'un tour et les journalise en une ligne."""

    name: str = "tour"
    stages: dict[str, float] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)
    _logger: logging.Logger = field(default_factory=lambda: get_logger("latence"))

    @contextmanager
    def stage(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.stages[stage] = round(self.stages.get(stage, 0.0) + elapsed_ms, 1)

    def record(self, stage: str, milliseconds: float) -> None:
        self.stages[stage] = round(self.stages.get(stage, 0.0) + milliseconds, 1)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000.0, 1)

    def emit(self, **fields: Any) -> None:
        total = self.total_ms
        livre = get_latency_book()
        for etage, valeur in self.stages.items():
            livre.record(etage.removesuffix("_ms"), valeur)
        livre.record("total", total)
        log_with(
            self._logger, logging.INFO, f"{self.name} terminé",
            total_ms=total, **self.stages, **fields,
        )

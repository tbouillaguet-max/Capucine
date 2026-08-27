"""Journalisation structurée et chronométrage par étage du pipeline.

L'objectif est le profilage sur Raspberry Pi : à la fin de chaque tour, une
ligne unique donne la latence de chaque étage (éveil, transcription,
inférence, exécution du plugin, synthèse) et le total.
"""

from __future__ import annotations

import json
import logging
import sys
import time
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
        log_with(
            self._logger, logging.INFO, f"{self.name} terminé",
            total_ms=self.total_ms, **self.stages, **fields,
        )

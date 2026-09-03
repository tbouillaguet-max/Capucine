"""Configuration en couches.

Ordre de priorité, du plus faible au plus fort :

1. ``config/default.toml`` livré avec le projet ;
2. le profil (``config/pc.toml`` ou ``config/pi.toml``), détecté ou imposé ;
3. les variables d'environnement ``LILY_<SECTION>__<CLE>`` ;
4. les options de ligne de commande.

``tomllib`` est dans la bibliothèque standard depuis Python 3.11 : la
configuration ne coûte aucune dépendance.
"""

from __future__ import annotations

import os
import platform
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .logging import get_logger

logger = get_logger("config")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ENV_PREFIX = "LILY_"


def detect_profile() -> str:
    """« pi » sur ARM/Linux non-Apple, « pc » partout ailleurs."""
    machine = platform.machine().lower()
    system = platform.system().lower()
    if system == "linux" and any(machine.startswith(p) for p in ("arm", "aarch")):
        return "pi"
    return "pc"


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _coerce_env(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in ("true", "vrai", "oui"):
        return True
    if lowered in ("false", "faux", "non"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if "," in raw:
        return [part.strip() for part in raw.split(",")]
    return raw


def _env_overlay(environ: Mapping[str, str]) -> dict[str, Any]:
    """``LILY_LLM__MODEL=qwen`` -> ``{"llm": {"model": "qwen"}}``.

    Le double tiret bas sépare les niveaux, un simple tiret bas reste dans le
    nom de la clé (``LILY_AUDIO__SAMPLE_RATE`` -> ``audio.sample_rate``).
    """
    overlay: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX) or name == ENV_PREFIX:
            continue
        path = [part.lower() for part in name[len(ENV_PREFIX):].split("__") if part]
        if not path:
            continue
        cursor = overlay
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):  # pragma: no cover - conflit improbable
                raise ConfigError(f"Variable d'environnement incohérente : {name}")
        cursor[path[-1]] = _coerce_env(raw)
    return overlay


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Fichier de configuration introuvable : {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"TOML invalide dans {path} : {exc}") from exc


class Config:
    """Vue en lecture seule sur la configuration fusionnée."""

    def __init__(self, data: Mapping[str, Any], sources: list[str] | None = None) -> None:
        self._data: dict[str, Any] = deepcopy(dict(data))
        self.sources = sources or []

    # -- lecture ------------------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        cursor: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                return default
            cursor = cursor[part]
        return deepcopy(cursor) if isinstance(cursor, (dict, list)) else cursor

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, {})
        return value if isinstance(value, dict) else {}

    def plugin_config(self, plugin_name: str, defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Config d'un plugin : ``CONFIG_DEFAULTS`` surchargé par
        ``[plugins.<nom>]``. Les plugins vivent sous ``plugins.`` pour qu'un
        plugin nommé ``audio`` ou ``llm`` n'écrase jamais une clé du cœur."""
        merged = dict(defaults or {})
        merged.update(self.section("plugins").get(plugin_name, {}) or {})
        return merged

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def with_overrides(self, overrides: Mapping[str, Any]) -> Config:
        return Config(_deep_merge(self._data, overrides), self.sources + ["overrides"])

    # -- chemins ------------------------------------------------------------
    def resolve_path(self, dotted_key: str, default: str | None = None) -> Path | None:
        raw = self.get(dotted_key, default)
        if raw in (None, ""):
            return None
        path = Path(str(raw)).expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def plugin_paths(self) -> list[Path]:
        raw = self.get("plugins.paths", ["./plugins"])
        if isinstance(raw, str):
            raw = [raw]
        paths: list[Path] = []
        for entry in raw:
            path = Path(str(entry)).expanduser()
            paths.append(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())
        return paths

    def __repr__(self) -> str:  # pragma: no cover - confort de débogage
        return f"Config(sources={self.sources!r})"


def load_config(
    profile: str | None = None,
    extra_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    config_dir: Path | None = None,
) -> Config:
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    environ = os.environ if environ is None else environ

    data = _read_toml(directory / "default.toml")
    sources = [str(directory / "default.toml")]

    chosen = profile or environ.get(f"{ENV_PREFIX}PROFILE") or detect_profile()
    profile_path = directory / f"{chosen}.toml"
    if profile_path.exists():
        data = _deep_merge(data, _read_toml(profile_path))
        sources.append(str(profile_path))
    else:
        logger.warning("Profil « %s » sans fichier %s, on garde les défauts.", chosen, profile_path.name)

    if extra_file:
        extra_path = Path(extra_file).expanduser()
        data = _deep_merge(data, _read_toml(extra_path))
        sources.append(str(extra_path))

    env_overlay = _env_overlay(environ)
    if env_overlay:
        data = _deep_merge(data, env_overlay)
        sources.append("env")

    if overrides:
        data = _deep_merge(data, {k: v for k, v in overrides.items() if v is not None})
        sources.append("cli")

    data.setdefault("profile", chosen)
    data["profile"] = chosen
    return Config(data, sources)

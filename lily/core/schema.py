"""Génération du schéma d'outil à partir d'une signature Python.

C'est la promesse centrale du contrat de plugin : le développeur écrit une
fonction annotée avec une docstring, et le cœur en déduit le JSON que le LLM
consommera. Personne n'écrit de JSON à la main.

Le module fait aussi le chemin inverse — la **coercition** — parce qu'un modèle
7-8B quantifié répond ``{"faces": "vingt"}`` là où la signature demande un
``int``. Refuser serait techniquement correct et pratiquement inutilisable.
"""

from __future__ import annotations

import inspect
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, get_args, get_origin

from .errors import ArgumentError, SchemaError
from .logging import get_logger
from .text import normalize, parse_french_number

logger = get_logger("schema")

__all__ = [
    "ParameterSpec",
    "resolve_hints",
    "build_parameters_schema",
    "build_tool_schema",
    "coerce_arguments",
    "parse_docstring",
]

_TRUE_WORDS = {"true", "vrai", "oui", "yes", "1", "on", "actif"}
_FALSE_WORDS = {"false", "faux", "non", "no", "0", "off", "inactif"}

_PARAM_PATTERNS = (
    re.compile(r"^\s*:param\s+(?P<name>\w+)\s*:\s*(?P<desc>.+)$"),
    re.compile(r"^\s*(?P<name>\w+)\s*(?:\([^)]*\))?\s*:\s*(?P<desc>.+)$"),
)
_ARGS_HEADERS = ("args:", "arguments:", "params:", "parameters:", "paramètres:", "parametres:")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Sépare le résumé de la docstring des descriptions de paramètres.

    Comprend le style reST (``:param x:``) et le style Google (section
    ``Args:``). Le reste de la docstring devient le contexte donné au LLM.
    """
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in _ARGS_HEADERS:
            in_args = True
            continue
        if in_args and stripped and not line.startswith((" ", "\t")) and stripped.endswith(":"):
            in_args = False
        match = _PARAM_PATTERNS[0].match(line)
        if match:
            params[match.group("name")] = match.group("desc").strip()
            continue
        if in_args:
            match = _PARAM_PATTERNS[1].match(line)
            if match:
                params[match.group("name")] = match.group("desc").strip()
                continue
            if stripped:
                continue
        if not in_args:
            summary.append(line)
    return "\n".join(summary).strip(), params


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    annotation: Any
    default: Any
    required: bool
    description: str
    json_schema: dict[str, Any]


def _unwrap_annotated(annotation: Any) -> tuple[Any, str | None]:
    if get_origin(annotation) is typing.Annotated:
        args = get_args(annotation)
        extra_doc = next((a for a in args[1:] if isinstance(a, str)), None)
        return args[0], extra_doc
    return annotation, None


def _optional_inner(annotation: Any) -> tuple[Any, bool]:
    """``str | None`` -> ``(str, True)``."""
    origin = get_origin(annotation)
    if origin is typing.Union or origin is getattr(__import__("types"), "UnionType", None):
        args = [a for a in get_args(annotation) if a is not type(None)]
        nullable = len(args) != len(get_args(annotation))
        if len(args) == 1:
            return args[0], nullable
        return annotation, nullable
    return annotation, False


def _json_type(annotation: Any, default: Any) -> dict[str, Any]:
    annotation, _ = _unwrap_annotated(annotation)
    annotation, nullable = _optional_inner(annotation)

    if annotation is inspect.Parameter.empty:
        # Pas d'annotation : on devine d'après la valeur par défaut, sinon
        # une chaîne, qui est ce que produit une transcription vocale.
        annotation = type(default) if default not in (None, inspect.Parameter.empty) else str

    schema: dict[str, Any]
    origin = get_origin(annotation)

    if origin is Literal:
        options = list(get_args(annotation))
        kinds = {type(o) for o in options}
        kind = "string" if kinds != {int} else "integer"
        schema = {"type": kind, "enum": options}
    elif isinstance(annotation, type) and issubclass(annotation, Enum):
        schema = {"type": "string", "enum": [member.value for member in annotation]}
    elif annotation is bool:
        schema = {"type": "boolean"}
    elif annotation is int:
        schema = {"type": "integer"}
    elif annotation is float:
        schema = {"type": "number"}
    elif annotation is str:
        schema = {"type": "string"}
    elif origin in (list, set, tuple) or annotation in (list, set, tuple):
        args = get_args(annotation)
        item = _json_type(args[0], None) if args else {"type": "string"}
        schema = {"type": "array", "items": item}
    elif origin is dict or annotation is dict:
        schema = {"type": "object"}
    elif annotation is type(None):
        schema = {"type": "null"}
    else:
        raise SchemaError(
            f"Type non pris en charge dans une signature de skill : {annotation!r}. "
            "Utilisez str, int, float, bool, list[...], dict, Literal[...] ou Enum."
        )

    if nullable:
        schema = {**schema, "nullable": True}
    return schema


def resolve_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Résout les annotations, y compris quand elles sont des chaînes.

    Beaucoup de fichiers commencent par ``from __future__ import annotations``,
    et un plugin a parfaitement le droit de le faire : ses annotations arrivent
    alors sous forme de texte (``"int"``). Sans cette résolution, le schéma
    d'outil serait faux pour la moitié des plugins réels.
    """
    try:
        return typing.get_type_hints(func, include_extras=True)
    except Exception:  # noqa: BLE001 - une annotation exotique ne casse rien
        pass
    # Résolution au cas par cas : une seule annotation illisible ne doit pas
    # faire perdre toutes les autres.
    resolved: dict[str, Any] = {}
    globalns = getattr(func, "__globals__", {})
    localns = vars(typing)
    for name, annotation in getattr(func, "__annotations__", {}).items():
        if not isinstance(annotation, str):
            resolved[name] = annotation
            continue
        try:
            resolved[name] = eval(annotation, globalns, localns)  # noqa: S307
        except Exception:  # noqa: BLE001
            logger.debug("Annotation illisible ignorée : %s: %s", name, annotation)
    return resolved


def build_parameters_schema(func: Callable[..., Any]) -> tuple[dict[str, Any], list[ParameterSpec]]:
    """Traduit la signature en JSON Schema d'objet + specs internes."""
    signature = inspect.signature(func)
    hints = resolve_hints(func)
    _, doc_params = parse_docstring(func.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []
    specs: list[ParameterSpec] = []

    for name, parameter in signature.parameters.items():
        if name in ("self", "cls"):
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise SchemaError(
                f"Le skill « {func.__name__} » utilise *args/**kwargs : impossible d'en "
                "déduire un schéma d'outil. Déclarez des paramètres nommés."
            )
        annotation = hints.get(name, parameter.annotation)
        if isinstance(annotation, str):  # non résolue : on la traite comme absente
            annotation = inspect.Parameter.empty
        _, annotated_doc = _unwrap_annotated(annotation)
        description = doc_params.get(name) or annotated_doc or ""
        json_schema = _json_type(annotation, parameter.default)
        if description:
            json_schema = {**json_schema, "description": description}
        is_required = parameter.default is inspect.Parameter.empty
        if not is_required and parameter.default is not None:
            json_schema = {**json_schema, "default": parameter.default}
        properties[name] = json_schema
        if is_required:
            required.append(name)
        specs.append(
            ParameterSpec(
                name=name,
                annotation=annotation,
                default=None if is_required else parameter.default,
                required=is_required,
                description=description,
                json_schema=json_schema,
            )
        )

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return schema, specs


def build_tool_schema(
    func: Callable[..., Any],
    *,
    name: str,
    description: str = "",
    examples: list[str] | None = None,
) -> dict[str, Any]:
    """Schéma d'outil au format OpenAI, compris par Ollama comme par llama.cpp."""
    parameters, _ = build_parameters_schema(func)
    doc_summary, _ = parse_docstring(func.__doc__)

    parts = [part for part in (description.strip(), doc_summary) if part]
    full_description = "\n\n".join(parts) or f"Exécute {name}."
    if examples:
        formulations = " ; ".join(f"« {example} »" for example in examples)
        full_description += f"\nFormulations typiques : {formulations}."

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": full_description,
            "parameters": parameters,
        },
    }


# --- coercition ------------------------------------------------------------

def _coerce_value(value: Any, schema: dict[str, Any], name: str) -> Any:
    if value is None:
        if schema.get("nullable"):
            return None
        raise ArgumentError(f"L'argument « {name} » ne peut pas être vide.")

    kind = schema.get("type", "string")
    enum = schema.get("enum")

    if enum is not None:
        for option in enum:
            if value == option or normalize(str(value)) == normalize(str(option)):
                return option
        allowed = ", ".join(str(o) for o in enum)
        raise ArgumentError(f"« {name} » doit valoir l'une de ces valeurs : {allowed}.")

    if kind == "boolean":
        if isinstance(value, bool):
            return value
        lowered = normalize(str(value))
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
        raise ArgumentError(f"« {name} » attend oui ou non, reçu {value!r}.")

    if kind in ("integer", "number"):
        if isinstance(value, bool):
            raise ArgumentError(f"« {name} » attend un nombre, reçu un booléen.")
        number = value if isinstance(value, (int, float)) else parse_french_number(str(value))
        if number is None:
            raise ArgumentError(f"« {name} » attend un nombre, reçu {value!r}.")
        if kind == "integer":
            if isinstance(number, float) and not number.is_integer():
                raise ArgumentError(f"« {name} » attend un entier, reçu {value!r}.")
            return int(number)
        return float(number)

    if kind == "array":
        items_schema = schema.get("items", {"type": "string"})
        if isinstance(value, str):
            raw_items = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raw_items = [value]
        return [_coerce_value(item, items_schema, f"{name}[]") for item in raw_items]

    if kind == "object":
        if isinstance(value, dict):
            return value
        raise ArgumentError(f"« {name} » attend un objet, reçu {value!r}.")

    return value if isinstance(value, str) else str(value)


def coerce_arguments(
    parameters_schema: dict[str, Any],
    raw_arguments: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Valide et convertit les arguments proposés par le LLM.

    Retourne ``(arguments_propres, clés_ignorées)``. Les clés inventées par le
    modèle sont écartées plutôt que fatales : un argument en trop ne doit pas
    faire échouer une commande par ailleurs correcte.
    """
    raw_arguments = dict(raw_arguments or {})
    properties: dict[str, Any] = parameters_schema.get("properties", {})
    required: list[str] = list(parameters_schema.get("required", []))

    cleaned: dict[str, Any] = {}
    dropped = [key for key in raw_arguments if key not in properties]

    for key, schema in properties.items():
        if key in raw_arguments:
            cleaned[key] = _coerce_value(raw_arguments[key], schema, key)
        elif key in required:
            raise ArgumentError(f"Il manque l'argument « {key} ».")
    return cleaned, dropped

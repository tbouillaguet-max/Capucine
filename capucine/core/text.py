"""Normalisation de texte et lecture des nombres écrits en toutes lettres.

Utilisé à deux endroits du cœur :

* le routeur, pour comparer une phrase de l'utilisateur aux ``examples`` d'un
  skill sans se soucier des accents ni de la ponctuation ;
* la coercition d'arguments, parce qu'un modèle local répond volontiers
  ``{"faces": "vingt"}`` là où le plugin annonce ``faces: int``.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "strip_accents",
    "normalize",
    "tokenize",
    "ascii_identifier",
    "parse_french_number",
    "extract_numbers",
    "similarity",
]

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Minuscules, sans accent, sans ponctuation, espaces compactés."""
    lowered = strip_accents(text).lower()
    return _SPACES.sub(" ", _PUNCT.sub(" ", lowered)).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize(text)
    return normalized.split() if normalized else []


def ascii_identifier(name: str) -> str:
    """Nom d'outil exposé au LLM : ASCII pur, car tous les modèles ne
    tokenisent pas proprement ``lancer_dé``. Les accents restent dans la
    description et les exemples, qui eux sont lus par le modèle comme du texte.
    """
    ascii_name = strip_accents(name)
    ascii_name = re.sub(r"[^0-9a-zA-Z_]", "_", ascii_name)
    ascii_name = re.sub(r"_+", "_", ascii_name).strip("_")
    if not ascii_name:
        ascii_name = "outil"
    if ascii_name[0].isdigit():
        ascii_name = f"_{ascii_name}"
    return ascii_name


# --- nombres français ------------------------------------------------------

_UNITS: dict[str, int] = {
    "zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4,
    "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
    "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15,
    "seize": 16, "trente": 30, "quarante": 40, "cinquante": 50,
    "soixante": 60,
}
_SCORE = {"vingt", "vingts"}
_HUNDRED = {"cent", "cents"}
_THOUSAND = {"mille", "milles"}
_SKIP = {"et", "-"}

_NUMBER_WORDS = set(_UNITS) | _SCORE | _HUNDRED | _THOUSAND | {"et"}
_DIGITS = re.compile(r"^-?\d+(?:[.,]\d+)?$")


def parse_french_number(text: str) -> int | float | None:
    """``"quatre-vingt-dix"`` -> ``90``. Retourne ``None`` si ce n'est pas un
    nombre. Accepte aussi les chiffres, et la virgule décimale française."""
    # Les chiffres sont testés avant `normalize`, qui mangerait la virgule
    # décimale française (« 3,5 »).
    raw = strip_accents(text).strip().lower().replace(" ", "")
    if _DIGITS.match(raw):
        raw = raw.replace(",", ".")
        return float(raw) if "." in raw else int(raw)

    candidate = normalize(text).replace("-", " ").strip()
    if not candidate:
        return None

    words = candidate.split()
    if not words or any(w not in _NUMBER_WORDS for w in words):
        return None

    total = 0
    current = 0
    seen = False
    for word in words:
        if word in _SKIP:
            continue
        seen = True
        if word in _SCORE:
            # « quatre-vingt » multiplie, « vingt » seul additionne.
            current = current * 20 if 2 <= current <= 9 else current + 20
        elif word in _HUNDRED:
            current = (current or 1) * 100
        elif word in _THOUSAND:
            total += (current or 1) * 1000
            current = 0
        else:
            current += _UNITS[word]
    return total + current if seen else None


def extract_numbers(text: str) -> list[int | float]:
    """Tous les nombres d'une phrase, en chiffres ou en toutes lettres."""
    tokens = normalize(text).replace("-", " ").split()
    found: list[int | float] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            # « un » / « une » isolés sont presque toujours des articles en
            # français (« lance un dé à vingt faces » ne contient qu'un
            # nombre). On préfère en rater un que d'en inventer un.
            if buffer not in (["un"], ["une"]):
                value = parse_french_number(" ".join(buffer))
                if value is not None:
                    found.append(value)
            buffer.clear()

    for token in tokens:
        if token in _NUMBER_WORDS:
            # « et » ne démarre jamais un nombre : « pain et vingt » .
            if token == "et" and not buffer:
                continue
            buffer.append(token)
            continue
        flush()
        if _DIGITS.match(token):
            value = parse_french_number(token)
            if value is not None:
                found.append(value)
    flush()
    return found


def similarity(left: str, right: str) -> float:
    """Score 0..1 entre deux phrases : recouvrement de tokens (F1) mêlé à une
    similarité de caractères, pour rattraper les variantes morphologiques.
    """
    from difflib import SequenceMatcher

    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    shared = len(left_tokens & right_tokens)
    if shared:
        precision = shared / len(right_tokens)
        recall = shared / len(left_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    ratio = SequenceMatcher(None, normalize(left), normalize(right)).ratio()
    return 0.7 * f1 + 0.3 * ratio

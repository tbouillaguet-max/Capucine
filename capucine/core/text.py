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
from collections.abc import Iterable, Iterator

__all__ = [
    "accord_ou_refus",
    "split_sentences",
    "stream_sentences",
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


# --- découpage en phrases --------------------------------------------------

# La synthèse se fait phrase par phrase : c'est l'unité que Piper produit, et
# celle à laquelle on peut s'interrompre proprement. Encore faut-il ne pas
# couper au milieu de « M. Dupont » ou de « 3.5 ».
# Seuls les *titres* et abréviations qui introduisent toujours autre chose
# bloquent la coupure : « M. Dupont » n'est pas une fin de phrase. « etc. » en
# est une la plupart du temps, et une coupure ratée coûte moins cher qu'une
# coupure au milieu d'un nom propre.
_TITRES = {
    "m", "mm", "mme", "mlle", "mr", "dr", "pr", "st", "ste",
    "av", "bd", "no", "n", "art", "env",
}
_FIN_DE_PHRASE = ".!?\u2026"
_FERMANTS = '"\u00bb)]'


def _points_de_coupe(text: str) -> list[int]:
    """Indices exclusifs où une phrase se termine, dans le texte **brut**.

    Travailler sur des décalages plutôt que sur des morceaux découpés est ce
    qui permet au flux (``stream_sentences``) de ne perdre aucune espace.
    """
    coupes: list[int] = []
    i = 0
    longueur = len(text)
    while i < longueur:
        caractere = text[i]
        if caractere == "\n" and text[i + 1 : i + 2] == "\n":
            coupes.append(i + 1)
            i += 2
            continue
        if caractere in _FIN_DE_PHRASE:
            # « ... », « ?! » : une seule fin de phrase.
            while i + 1 < longueur and text[i + 1] in _FIN_DE_PHRASE:
                i += 1
            fin = i + 1
            # On emporte un guillemet ou une parenthèse fermante.
            while fin < longueur and text[fin] in _FERMANTS:
                fin += 1
            suivant = text[fin : fin + 1]
            if suivant in ("", " ", "\n", "\t") and (
                caractere != "." or _fin_reelle(text, i, text[i + 1 : i + 2])
            ):
                coupes.append(fin)
                i = fin
                continue
        i += 1
    return coupes


def _fin_reelle(text: str, position: int, suivant: str) -> bool:
    """Un point termine-t-il vraiment la phrase ?"""
    precedent = text[position - 1 : position]
    if precedent.isdigit() and suivant[:1].isdigit():
        return False  # « 3.5 »
    debut = position - 1
    while debut >= 0 and (text[debut].isalpha() or text[debut] == "'"):
        debut -= 1
    mot = strip_accents(text[debut + 1 : position]).lower()
    return mot not in _TITRES  # « M. Dupont »


def split_sentences(text: str, min_chars: int = 1) -> list[str]:
    """Découpe un texte en phrases, à la française.

    Args:
        text: Le texte à découper.
        min_chars: Longueur en dessous de laquelle un fragment est recollé au
            précédent, pour ne pas payer une synthèse entière pour « Ah ».
    """
    coupes = [*_points_de_coupe(text), len(text)]
    morceaux: list[str] = []
    precedent = 0
    for coupe in coupes:
        if coupe > precedent:
            morceaux.append(text[precedent:coupe])
            precedent = coupe

    resultat: list[str] = []
    attente = ""
    for morceau in (m.strip() for m in morceaux):
        if not morceau:
            continue
        if attente:
            morceau = f"{attente} {morceau}"
            attente = ""
        if len(morceau) < min_chars or not any(c.isalnum() for c in morceau):
            # Trop court pour mériter sa propre synthèse : on le recolle au
            # précédent s'il y en a un, sinon au suivant.
            if resultat:
                resultat[-1] = f"{resultat[-1]} {morceau}"
            else:
                attente = morceau
            continue
        resultat.append(morceau)
    if attente:
        resultat.append(attente)
    return resultat


def stream_sentences(morceaux: Iterable[str], min_chars: int = 1) -> Iterator[str]:
    """Transforme un flux de fragments en flux de phrases complètes.

    C'est ce qui permet de commencer à parler pendant que le modèle écrit
    encore : dès qu'une phrase est terminée, elle part à la synthèse, et la
    première parole sort bien avant la fin de l'inférence.
    """
    tampon = ""
    attente = ""
    for morceau in morceaux:
        if not morceau:
            continue
        tampon += morceau
        coupes = _points_de_coupe(tampon)
        if not coupes:
            continue
        precedent = 0
        for coupe in coupes:
            phrase = tampon[precedent:coupe].strip()
            precedent = coupe
            if not phrase:
                continue
            phrase = f"{attente} {phrase}".strip() if attente else phrase
            if len(phrase) < min_chars or not any(c.isalnum() for c in phrase):
                attente = phrase
                continue
            attente = ""
            yield phrase
        tampon = tampon[precedent:]

    reste = f"{attente} {tampon}".strip() if attente else tampon.strip()
    if reste:
        yield reste


# --- oui, non, ou ni l'un ni l'autre ---------------------------------------

_ACCORDS = {
    "oui", "ouais", "ouaip", "si", "d accord", "daccord", "vas y", "allez y",
    "confirme", "je confirme", "fais le", "faites le", "bien sur", "exact",
    "c est ca", "affirmatif", "ok", "okay", "parfait", "valide",
}
_REFUS = {
    "non", "nan", "surtout pas", "annule", "annuler", "laisse tomber",
    "laissez tomber", "arrete", "arretez", "negatif", "pas du tout",
    "non merci", "oublie", "oubliez",
}
_MOTS_ACCORD = {mot for accord in _ACCORDS for mot in accord.split()}
_MOTS_REFUS = {mot for refus in _REFUS for mot in refus.split()}


def accord_ou_refus(phrase: str) -> bool | None:
    """``True`` pour un oui, ``False`` pour un non, ``None`` si ça ne tranche pas.

    Volontairement strict : on juge la phrase entière, pas un mot au milieu.
    « Non, plutôt trois minutes » ne doit pas être lu comme un refus sec — c'est
    une nouvelle demande — et « oui mais avant, quelle heure est-il ? » non plus
    comme un accord. Dans le doute, on rend ``None`` et l'appelant repart en
    routage normal plutôt que de piéger l'utilisateur dans sa question.
    """
    normalisee = normalize(phrase)
    if not normalisee:
        return None
    if normalisee in _ACCORDS:
        return True
    if normalisee in _REFUS:
        return False
    # « oui vas-y », « non annule » : plusieurs marqueurs, tous du même bord.
    mots = set(normalisee.split())
    if not mots:
        return None
    if mots <= _MOTS_ACCORD:
        return True
    if mots <= _MOTS_REFUS:
        return False
    return None

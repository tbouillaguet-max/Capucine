"""Lecture simple : le plus petit plugin utile.

Ce qu'il montre :

* le contrat minimal — un fichier, un import, un décorateur ;
* le retour ``{"speak": …, "display": …}``, qui dissocie ce qui est **dit** de
  ce qui est **journalisé**. « 9 h 20 » se lit mal à voix haute et « neuf
  heures vingt » se relit mal dans un journal : on donne les deux ;
* un paramètre ``Literal``, qui devient une énumération dans le schéma d'outil
  et interdit au modèle d'inventer une valeur.
"""

from datetime import datetime, timedelta
from typing import Literal

from lily.plugin import skill

_UNITES = [
    "zéro", "une", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
    "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "dix-sept", "dix-huit", "dix-neuf",
]
_DIZAINES = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante"}
_JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MOIS = [
    "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
    "août", "septembre", "octobre", "novembre", "décembre",
]


def _en_lettres(nombre: int) -> str:
    """0 à 59 en toutes lettres. Piper prononce « 9 h 20 » n'importe comment."""
    if nombre < 20:
        return _UNITES[nombre]
    dizaine, unite = divmod(nombre, 10)
    base = _DIZAINES[dizaine * 10]
    if unite == 0:
        return base
    if unite == 1:
        return f"{base} et une"
    return f"{base}-{_UNITES[unite]}"


@skill(
    description="Donne l'heure qu'il est.",
    examples=["quelle heure est-il", "il est quelle heure", "donne-moi l'heure"],
)
def heure() -> dict:
    """Lit l'heure courante de la machine.

    Aucun réseau, aucun serveur de temps : l'horloge système suffit, et elle
    fonctionne le Wi-Fi coupé.
    """
    maintenant = datetime.now()
    heures, minutes = maintenant.hour, maintenant.minute

    if minutes == 0:
        parle = f"Il est {_en_lettres(heures)} heure{'s' if heures > 1 else ''} pile."
    else:
        parle = f"Il est {_en_lettres(heures)} heure{'s' if heures > 1 else ''} {_en_lettres(minutes)}."
    return {"speak": parle, "display": maintenant.strftime("%H:%M")}


@skill(
    description="Donne la date du jour, d'hier ou de demain.",
    examples=["on est quel jour", "quelle est la date", "on est le combien demain"],
)
def date(jour: Literal["aujourd'hui", "hier", "demain"] = "aujourd'hui") -> dict:
    """Lit la date à partir de l'horloge système.

    Args:
        jour: Le jour voulu, relatif à aujourd'hui.
    """
    decalage = {"hier": -1, "aujourd'hui": 0, "demain": 1}[jour]
    quand = datetime.now() + timedelta(days=decalage)
    numero = "premier" if quand.day == 1 else str(quand.day)
    parle = f"Nous sommes le {_JOURS[quand.weekday()]} {numero} {_MOIS[quand.month - 1]}."
    return {"speak": parle, "display": quand.strftime("%Y-%m-%d")}

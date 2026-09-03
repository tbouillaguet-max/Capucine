"""Tâche de fond avec état : le cas le plus délicat du contrat.

Ce qu'il montre :

* de l'**état en mémoire** entre deux appels — impossible avec
  ``@skill(isolate=True)``, et c'est pour cela que l'isolation en
  sous-processus n'est pas le défaut ;
* ``announce()`` : quand un minuteur arrive à échéance, il n'a personne à qui
  répondre. Il doit **interrompre** ;
* ``on_unload()`` : sans lui, chaque rechargement à chaud laisserait derrière
  lui des minuteurs orphelins qui sonneraient dans le vide ;
* plusieurs compétences dans un seul fichier.
"""

import math
import threading
import time

from lily.plugin import announce, get_logger, skill

CONFIG_DEFAULTS = {
    "duree_max_minutes": 180,
    "maximum_simultane": 8,
}

# L'état vit ici, dans le module. Le registre remplace le module entier au
# rechargement, donc cet état disparaît proprement — à condition que
# on_unload() ait annulé les minuteries.
_MINUTEURS: dict[str, dict] = {}
_VERROU = threading.Lock()


def on_unload() -> None:
    """Annule tout avant que le module ne disparaisse.

    C'est le point le plus important du fichier. Une ``threading.Timer`` qui
    survit à son module continue de tourner et finit par appeler du code
    déchargé : au mieux elle sonne pour rien, au pire elle lève dans un thread
    que plus personne ne surveille.
    """
    with _VERROU:
        for minuteur in _MINUTEURS.values():
            minuteur["timer"].cancel()
        nombre = len(_MINUTEURS)
        _MINUTEURS.clear()
    if nombre:
        get_logger().info("%d minuteur(s) annulé(s) au déchargement.", nombre)


def _libelle_libre(souhaite: str) -> str:
    base = souhaite.strip() or "minuteur"
    if base not in _MINUTEURS:
        return base
    index = 2
    while f"{base} {index}" in _MINUTEURS:
        index += 1
    return f"{base} {index}"


def _duree_parlee(secondes: int) -> str:
    minutes, reste = divmod(int(secondes), 60)
    if minutes and reste:
        return f"{minutes} minute{'s' if minutes > 1 else ''} et {reste} seconde{'s' if reste > 1 else ''}"
    if minutes:
        return f"{minutes} minute{'s' if minutes > 1 else ''}"
    return f"{reste} seconde{'s' if reste > 1 else ''}"


def _sonner(libelle: str) -> None:
    """Appelé depuis le thread de la minuterie, hors de tout tour de parole."""
    with _VERROU:
        _MINUTEURS.pop(libelle, None)
    announce(f"C'est l'heure : {libelle}.")


@skill(
    description="Lance un minuteur qui préviendra à voix haute quand il sonne.",
    examples=[
        "mets un minuteur de dix minutes",
        "réveille-moi dans un quart d'heure",
        "minuteur de trois minutes pour les pâtes",
    ],
)
def minuteur(minutes: int = 0, secondes: int = 0, libelle: str = "minuteur") -> str:
    """Programme une alerte vocale à échéance.

    Args:
        minutes: Nombre de minutes à attendre.
        secondes: Secondes supplémentaires.
        libelle: À quoi sert ce minuteur, pour le retrouver et l'annoncer.
    """
    from lily.plugin import get_config

    total = int(minutes) * 60 + int(secondes)
    if total <= 0:
        return "Il me faut une durée : dix minutes, trente secondes…"

    maximum = int(get_config("duree_max_minutes", 180)) * 60
    if total > maximum:
        return f"C'est trop long pour moi, je ne dépasse pas {maximum // 60} minutes."

    with _VERROU:
        if len(_MINUTEURS) >= int(get_config("maximum_simultane", 8)):
            return "J'ai déjà trop de minuteurs en cours."
        nom = _libelle_libre(libelle)
        timer = threading.Timer(total, _sonner, args=(nom,))
        timer.daemon = True
        _MINUTEURS[nom] = {"timer": timer, "echeance": time.time() + total, "duree": total}
        timer.start()

    return f"C'est parti pour {_duree_parlee(total)}."


@skill(
    description="Dit quels minuteurs sont en cours et combien de temps il leur reste.",
    examples=["où en sont mes minuteurs", "combien de temps il reste", "mes minuteurs"],
)
def minuteurs() -> dict:
    """Liste les minuteurs en cours."""
    with _VERROU:
        en_cours = [
            (nom, max(0, math.ceil(donnees["echeance"] - time.time())))
            for nom, donnees in _MINUTEURS.items()
        ]
    if not en_cours:
        return {"speak": "Aucun minuteur en cours.", "display": "0 minuteur"}

    morceaux = [f"{nom}, {_duree_parlee(restant)}" for nom, restant in en_cours]
    return {
        "speak": "Il reste " + " ; ".join(morceaux) + ".",
        "display": " | ".join(f"{nom}={restant}s" for nom, restant in en_cours),
    }


@skill(
    description="Annule un minuteur en cours, ou tous.",
    examples=["annule le minuteur", "arrête le minuteur des pâtes", "annule tous les minuteurs"],
)
def annuler_minuteur(libelle: str = "") -> str:
    """Annule un minuteur.

    Args:
        libelle: Le minuteur à annuler. Vide, ils sont tous annulés.
    """
    with _VERROU:
        if not _MINUTEURS:
            return "Il n'y a aucun minuteur à annuler."
        cible = libelle.strip().lower()
        if not cible:
            for minuteur_en_cours in _MINUTEURS.values():
                minuteur_en_cours["timer"].cancel()
            nombre = len(_MINUTEURS)
            _MINUTEURS.clear()
            return f"{nombre} minuteur{'s' if nombre > 1 else ''} annulé{'s' if nombre > 1 else ''}."

        trouve = next((nom for nom in _MINUTEURS if cible in nom.lower()), None)
        if trouve is None:
            return f"Je ne trouve pas de minuteur nommé « {libelle} »."
        _MINUTEURS.pop(trouve)["timer"].cancel()
        return f"{trouve} annulé."

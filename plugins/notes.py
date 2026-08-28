"""Persistance sur disque : garder quelque chose entre deux démarrages.

Ce qu'il montre :

* ``data_dir()`` — un dossier inscriptible réservé au plugin, créé à la
  demande. Un plugin n'écrit jamais où bon lui semble ;
* ``CONFIG_DEFAULTS`` + ``get_config()`` — des réglages surchargeables depuis
  ``[plugins.notes]`` du fichier de configuration ;
* ``@skill(confirm=…)`` — une action irréversible n'est pas exécutée du
  premier coup : Capucine pose la question et attend un oui.

Le format est du JSON Lines : une note par ligne. Un fichier tronqué par une
coupure de courant ne perd que sa dernière ligne, là où un JSON unique serait
entièrement illisible.
"""

import json
from datetime import datetime

from capucine.plugin import data_dir, get_config, get_logger, skill

CONFIG_DEFAULTS = {
    "fichier": "notes.jsonl",
    "longueur_max": 500,
    "lecture_par_defaut": 3,
}


def _chemin():
    return data_dir() / str(get_config("fichier", "notes.jsonl"))


def _lire() -> list[dict]:
    chemin = _chemin()
    if not chemin.exists():
        return []
    notes = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            notes.append(json.loads(ligne))
        except json.JSONDecodeError:
            # Ligne tronquée par une coupure : on la saute, on garde le reste.
            get_logger().warning("Note illisible ignorée.")
    return notes


@skill(
    description="Prend une note et la garde pour plus tard.",
    examples=[
        "note que je dois appeler le plombier",
        "prends une note : acheter du pain",
        "rappelle-moi d'arroser les plantes",
    ],
)
def noter(texte: str) -> str:
    """Ajoute une note datée.

    Args:
        texte: Ce qu'il faut retenir, tel quel.
    """
    texte = texte.strip()
    if not texte:
        return "Il me faut quelque chose à noter."
    longueur_max = int(get_config("longueur_max", 500))
    if len(texte) > longueur_max:
        return f"C'est trop long pour une note, restez sous {longueur_max} caractères."

    note = {"date": datetime.now().isoformat(timespec="seconds"), "texte": texte}
    with _chemin().open("a", encoding="utf-8") as fichier:
        fichier.write(json.dumps(note, ensure_ascii=False) + "\n")
    return "C'est noté."


@skill(
    description="Relit les dernières notes prises.",
    examples=["relis mes notes", "qu'est-ce que j'ai noté", "mes dernières notes"],
)
def mes_notes(nombre: int = 0) -> dict:
    """Relit les notes les plus récentes.

    Args:
        nombre: Combien de notes relire. Zéro prend la valeur configurée.
    """
    notes = _lire()
    if not notes:
        return {"speak": "Vous n'avez aucune note.", "display": "0 note"}

    combien = int(nombre) or int(get_config("lecture_par_defaut", 3))
    dernieres = notes[-max(1, combien):]
    parle = " ".join(f"{note['texte']}." for note in dernieres)
    return {
        "speak": parle,
        "display": "\n".join(f"{note['date']} — {note['texte']}" for note in dernieres),
    }


@skill(
    description="Cherche une note contenant un mot.",
    examples=["retrouve ma note sur le plombier", "cherche dans mes notes le mot pain"],
)
def chercher_note(mot: str) -> dict:
    """Cherche parmi les notes.

    Args:
        mot: Le mot ou la phrase à retrouver.
    """
    mot = mot.strip().lower()
    trouvees = [note for note in _lire() if mot in note["texte"].lower()]
    if not trouvees:
        return {"speak": f"Rien sur « {mot} ».", "display": f"0 résultat pour {mot!r}"}
    parle = " ".join(f"{note['texte']}." for note in trouvees[-3:])
    return {"speak": parle, "display": f"{len(trouvees)} résultat(s) pour {mot!r}"}


@skill(
    description="Efface toutes les notes, définitivement.",
    examples=["efface toutes mes notes", "supprime mes notes"],
    confirm="Voulez-vous vraiment effacer toutes vos notes ?",
)
def effacer_notes() -> str:
    """Supprime le fichier de notes.

    Déclarée ``confirm=`` : Capucine pose la question et n'exécute qu'après un
    oui. Une commande vocale mal transcrite ne doit pas détruire des données.
    """
    chemin = _chemin()
    nombre = len(_lire())
    chemin.unlink(missing_ok=True)
    return f"{nombre} note{'s' if nombre > 1 else ''} effacée{'s' if nombre > 1 else ''}."

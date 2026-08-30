"""Dire ce qu'elle sait faire, en listant ce qui le lui apprend.

Une capacité, dans ce projet, **c'est un fichier** dans ``plugins/`` (voir
``routines.py``). Lister ses capacités, c'est donc lister ce dossier-là — pas
fouiller l'atelier de l'utilisateur, qui reste fermé par défaut
(``atelier.racines = []``) et n'a rien à voir avec ce que Capucine sait faire.

Sans cette compétence dédiée, « liste tes capacités » atterrissait sur
``chercher_dans_fichiers`` — une compétence de l'atelier — et échouait tant
qu'aucun dossier de travail n'était ouvert, alors que la question ne portait
sur aucun dossier de travail.
"""

from capucine.plugin import dossier_des_plugins, skill


@skill(
    description="Liste les fichiers de plugins qui donnent ses capacités à Capucine.",
    examples=[
        "liste tes capacités",
        "quelles sont tes capacités",
        "montre-moi tes capacités",
        "liste tes compétences",
        "qu'est-ce que tu sais faire",
    ],
)
def lister_mes_capacites() -> dict:
    """Liste le dossier plugins/ : chaque fichier y est une capacité."""
    fichiers = sorted(
        fichier.name for fichier in dossier_des_plugins().glob("*.py")
        if not fichier.name.startswith("_")
    )
    if not fichiers:
        return {
            "speak": "Je n'ai aucun fichier de capacité pour l'instant.",
            "display": "aucun fichier dans plugins/",
        }
    noms = [fichier[:-3] for fichier in fichiers]
    return {
        "speak": f"J'ai {len(fichiers)} fichiers de capacités : " + ", ".join(noms) + ".",
        "display": "\n".join(fichiers),
    }

"""Assistance sur vos fichiers, dans un périmètre que vous décidez.

**Rien n'est accessible tant que vous n'avez pas ouvert un dossier** dans
``[atelier] racines``. Ce n'est pas de la prudence excessive : la commande
arrive par la voix, une transcription est imparfaite, et un modèle 7B choisit
parfois mal ses arguments. Une capacité livrée inerte est une capacité qui ne
détruit rien avant que vous l'ayez voulue.

Les garanties, tenues par ``lily.core.atelier`` :

* tout chemin est résolu (liens symboliques compris) puis vérifié comme
  appartenant à une racine autorisée ;
* les identifiants et les clés restent hors de portée même dans une racine ;
* toute réécriture laisse une sauvegarde horodatée à côté du fichier ;
* **rien n'est supprimé** : les fichiers partent à la corbeille ;
* écrire, déplacer et jeter demandent confirmation à voix haute.
"""

from lily.plugin import atelier, get_config, skill

CONFIG_DEFAULTS = {
    "fichiers_listes": 25,
    "resultats_recherche": 12,
    "extrait_lignes": 40,
}


def _relatif(chemin) -> str:
    """Chemin lisible, relatif à la première racine de l'atelier."""
    for racine in atelier().racines:
        if chemin == racine or chemin.is_relative_to(racine):
            return str(chemin.relative_to(racine))
    return str(chemin)


@skill(
    description="Liste les fichiers d'un dossier de travail.",
    examples=[
        "liste les fichiers du projet",
        "qu'est-ce qu'il y a dans le dossier backtest",
        "montre-moi les fichiers python",
    ],
)
def lister_fichiers(dossier: str = ".", motif: str = "*") -> dict:
    """Liste le contenu d'un dossier de l'atelier.

    Args:
        dossier: Le dossier à regarder, relatif à l'atelier.
        motif: Un filtre, par exemple « *.py ».
    """
    entrees = atelier().lister(dossier, motif)
    maximum = int(get_config("fichiers_listes", 25))
    if not entrees:
        return {"speak": "Ce dossier est vide.", "display": f"{dossier} : 0 entrée"}

    noms = [f"{e.name}/" if e.is_dir() else e.name for e in entrees[:maximum]]
    parle = f"{len(entrees)} entrées. " + ", ".join(noms[:8])
    if len(entrees) > 8:
        parle += ", et d'autres"
    return {"speak": parle + ".", "display": "\n".join(noms)}


@skill(
    description="Lit le contenu d'un fichier.",
    examples=["lis le fichier config point py", "montre-moi le contenu de notes", "ouvre ce fichier"],
)
def lire_fichier(chemin: str, lignes: int = 0) -> dict:
    """Lit un fichier texte de l'atelier.

    Args:
        chemin: Le fichier à lire, relatif à l'atelier.
        lignes: Nombre de lignes à lire. Zéro prend la valeur configurée.
    """
    contenu = atelier().lire(chemin)
    maximum = int(lignes) or int(get_config("extrait_lignes", 40))
    toutes = contenu.splitlines()
    extrait = "\n".join(toutes[:maximum])
    reste = len(toutes) - maximum

    parle = f"{len(toutes)} lignes. " + " ".join(toutes[:6])
    return {
        "speak": parle[:500],
        "display": extrait + (f"\n… et {reste} lignes de plus" if reste > 0 else ""),
    }


@skill(
    description="Cherche un texte dans les fichiers du projet.",
    examples=[
        "cherche le mot backtest dans les fichiers",
        "où est défini calcul dcf",
        "trouve les fichiers qui parlent de volatilité",
    ],
    timeout=30.0,
)
def chercher_dans_fichiers(texte: str, motif: str = "*.py") -> dict:
    """Cherche une chaîne dans les fichiers de l'atelier.

    Args:
        texte: Ce qu'il faut trouver.
        motif: Les fichiers à fouiller, par exemple « *.py » ou « * ».
    """
    espace = atelier()
    cible = texte.strip().lower()
    if not cible:
        return {"speak": "Que dois-je chercher ?", "display": "recherche vide"}

    maximum = int(get_config("resultats_recherche", 12))
    trouvailles: list[str] = []
    for racine in espace.racines:
        for fichier in sorted(racine.rglob(motif)):
            if len(trouvailles) >= maximum:
                break
            if not fichier.is_file() or any(p.startswith(".") for p in fichier.parts):
                continue
            try:
                espace._verifier_interdits(fichier)
                lignes = fichier.read_text(encoding="utf-8", errors="replace").splitlines()
            except (OSError, PermissionError):
                continue
            for numero, ligne in enumerate(lignes, 1):
                if cible in ligne.lower():
                    trouvailles.append(f"{_relatif(fichier)}:{numero}: {ligne.strip()[:100]}")
                    break

    if not trouvailles:
        return {"speak": f"Rien sur « {texte} ».", "display": f"0 résultat pour {texte!r}"}
    fichiers = {t.split(":", 1)[0] for t in trouvailles}
    parle = (
        f"{len(trouvailles)} occurrence{'s' if len(trouvailles) > 1 else ''} dans "
        f"{len(fichiers)} fichier{'s' if len(fichiers) > 1 else ''}, "
        f"dont {', '.join(list(fichiers)[:3])}."
    )
    return {"speak": parle, "display": "\n".join(trouvailles)}


@skill(
    description="Ajoute du texte à la fin d'un fichier, sans rien effacer.",
    examples=["ajoute une ligne au fichier de notes", "complète le fichier todo"],
)
def ajouter_au_fichier(chemin: str, contenu: str) -> str:
    """Complète un fichier. Opération non destructive, donc sans confirmation.

    Args:
        chemin: Le fichier à compléter.
        contenu: Le texte à ajouter.
    """
    cible = atelier().ecrire(chemin, contenu.rstrip("\n") + "\n", ajouter=True)
    return f"Ajouté à {cible.name}."


@skill(
    description="Écrit ou remplace le contenu d'un fichier.",
    examples=["écris ce texte dans le fichier", "remplace le contenu de brouillon"],
    confirm="Voulez-vous vraiment que je réécrive ce fichier ? Je garde une sauvegarde.",
)
def ecrire_fichier(chemin: str, contenu: str) -> str:
    """Remplace le contenu d'un fichier, après sauvegarde de l'ancien.

    Args:
        chemin: Le fichier à écrire.
        contenu: Le nouveau contenu, en entier.
    """
    cible = atelier().ecrire(chemin, contenu)
    return f"{cible.name} écrit. L'ancien contenu est à côté, en sauvegarde."


@skill(
    description="Renomme ou déplace un fichier.",
    examples=["renomme ce fichier", "déplace le brouillon dans archives"],
    confirm="Voulez-vous vraiment déplacer ce fichier ?",
)
def deplacer_fichier(source: str, destination: str) -> str:
    """Déplace un fichier dans l'atelier.

    Args:
        source: Le fichier à déplacer.
        destination: Son nouveau chemin.
    """
    arrivee = atelier().deplacer(source, destination)
    return f"Déplacé vers {arrivee.name}."


@skill(
    description="Met un fichier à la corbeille.",
    examples=["supprime ce fichier", "jette le brouillon", "efface ce fichier"],
    confirm="Voulez-vous vraiment mettre ce fichier à la corbeille ?",
)
def jeter_fichier(chemin: str) -> str:
    """Déplace un fichier à la corbeille. Il n'y a pas de suppression ici.

    Args:
        chemin: Le fichier à jeter.
    """
    destination = atelier().jeter(chemin)
    return f"À la corbeille. Récupérable dans {destination.parent}."


@skill(
    description="Dit quels dossiers Lily a le droit de lire et d'écrire.",
    examples=["quels dossiers peux-tu voir", "où as-tu le droit d'écrire", "ton atelier"],
)
def mon_atelier() -> dict:
    """Décrit le périmètre ouvert à Lily."""
    try:
        espace = atelier()
    except PermissionError as exc:
        return {"speak": str(exc), "display": "atelier fermé"}
    return {
        "speak": f"Je travaille dans {espace.decrire()}.",
        "display": espace.decrire(),
    }

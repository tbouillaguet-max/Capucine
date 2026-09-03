"""Ce que Lily a appris de vous, et comment le lui faire oublier.

Ce plugin ne fait rien apprendre : l'apprentissage se produit tout seul, au
fil des tours, dans le cœur. Il ouvre seulement une fenêtre dessus — voir,
corriger, oublier. Une mémoire qu'on ne peut pas inspecter est une mémoire à
laquelle on ne peut pas faire confiance.
"""

from lily.plugin import apprentissage, corpus, get_config, skill

CONFIG_DEFAULTS = {
    "exemples_montres": 6,
    "mots_montres": 15,
}


@skill(
    description="Dit ce que Lily a appris de votre façon de parler.",
    examples=[
        "qu'est-ce que tu as appris",
        "qu'as-tu retenu de ma façon de parler",
        "montre-moi ton apprentissage",
    ],
)
def ce_que_tu_as_appris() -> dict:
    """Résume les formulations retenues et le vocabulaire collecté."""
    magasin = apprentissage()
    stats = magasin.statistiques()
    if not stats["phrases"] and not stats["mots"]:
        return {
            "speak": "Je n'ai encore rien appris de particulier.",
            "display": "aucune formulation, aucun mot",
        }

    parle = (
        f"J'ai retenu {stats['phrases']} formulation"
        f"{'s' if stats['phrases'] > 1 else ''} pour {stats['competences']} "
        f"compétence{'s' if stats['competences'] > 1 else ''}, "
        f"et {stats['mots']} mot{'s' if stats['mots'] > 1 else ''} de vocabulaire."
    )

    maximum = int(get_config("exemples_montres", 6))
    lignes: list[str] = []
    for outil, phrases in magasin.phrases_par_outil().items():
        for retenue in phrases[:2]:
            lignes.append(
                f"{outil:<22} ← « {retenue.phrase} »  "
                f"(vu {retenue.confirmations}×, poids {retenue.poids:.2f})"
            )
        if len(lignes) >= maximum:
            break
    return {"speak": parle, "display": "\n".join(lignes) or parle}


@skill(
    description="Dit quels mots de vocabulaire Lily souffle à la transcription.",
    examples=["quel vocabulaire tu connais", "quels mots tu souffles à whisper", "ton vocabulaire"],
)
def mon_vocabulaire() -> dict:
    """Les noms propres collectés, ceux que la transcription écorcherait."""
    mots = apprentissage().vocabulaire(limite=int(get_config("mots_montres", 15)))
    if not mots:
        return {"speak": "Je n'ai pas encore collecté de vocabulaire.", "display": "0 mot"}
    return {
        "speak": "Je connais " + ", ".join(entree.mot for entree in mots[:8]) + ".",
        "display": "\n".join(
            f"{entree.mot} — vu {entree.occurrences}× ({entree.source})" for entree in mots
        ),
    }


@skill(
    description="Ajoute un mot au vocabulaire soufflé à la transcription.",
    examples=[
        "retiens le mot CalculRisque",
        "ajoute ce nom à ton vocabulaire",
        "apprends à écrire ce mot",
    ],
)
def retenir_ce_mot(mot: str) -> str:
    """Force un mot dans le vocabulaire, sans attendre qu'il soit repéré.

    Args:
        mot: Le nom propre ou le terme technique à retenir.
    """
    if apprentissage().retenir_mot(mot.strip(), source="dicte"):
        return f"« {mot.strip()} » ajouté à mon vocabulaire."
    return "Ce mot est trop court ou trop courant pour être utile."


@skill(
    description="Fait oublier à Lily ce qu'elle a appris d'une compétence ou d'un mot.",
    examples=[
        "oublie ce que tu as appris sur le minuteur",
        "efface ton apprentissage",
        "désapprends cette formulation",
    ],
    confirm="Voulez-vous vraiment que j'oublie cet apprentissage ?",
)
def oublier_l_apprentissage(sujet: str = "") -> str:
    """Efface des formulations ou des mots appris.

    Args:
        sujet: La compétence ou le mot concerné. Vide, tout est oublié.
    """
    magasin = apprentissage()
    if not sujet.strip():
        stats = magasin.statistiques()
        magasin.tout_oublier()
        return (
            f"J'ai tout oublié : {stats['phrases']} formulation(s) et "
            f"{stats['mots']} mot(s). Je repars de vos exemples d'origine."
        )
    phrases = magasin.oublier_outil(sujet.strip())
    mots = magasin.oublier_mot(sujet.strip())
    if not phrases and not mots:
        return f"Je n'avais rien appris sur « {sujet} »."
    return f"Oublié : {phrases} formulation(s) et {mots} mot(s)."


# --- le corpus d'éveil : apprendre votre voix -------------------------------

@skill(
    description="Dit combien d'extraits d'éveil Lily a gardés pour réapprendre votre voix.",
    examples=[
        "où en est ton corpus d'éveil",
        "combien de fois tu t'es réveillée pour rien",
        "état de tes enregistrements d'éveil",
    ],
)
def mon_corpus_d_eveil() -> dict:
    """Le décompte des vrais et faux déclenchements enregistrés."""
    magasin = corpus()
    if not magasin.actif:
        return {
            "speak": "Je ne garde aucun extrait sonore. Pour que j'apprenne votre "
                     "voix, mettez « actif = true » dans la section corpus.",
            "display": "corpus d'éveil éteint — [corpus] actif = false\n"
                       f"Il écrirait dans : {magasin.dossier}",
        }

    etat = magasin.etat()
    if not etat.total:
        return {
            "speak": "Je n'ai encore gardé aucun extrait d'éveil.",
            "display": f"corpus vide — {etat.dossier}",
        }
    return {
        "speak": f"J'ai gardé {etat.eveils} éveil{'s' if etat.eveils > 1 else ''} "
                 f"et {etat.faux_positifs} déclenchement"
                 f"{'s' if etat.faux_positifs > 1 else ''} pour rien.",
        "display": (
            f"{etat.eveils} vrais éveils\n"
            f"{etat.faux_positifs} faux positifs "
            f"({etat.taux_de_faux_positifs:.0%} des déclenchements)\n"
            f"{etat.en_attente} en attente d'étiquetage\n"
            f"dossier : {etat.dossier}\n\n"
            "Pour réentraîner : python tools/entrainer_lily.py entrainer "
            "--corpus " + str(etat.dossier)
        ),
    }


@skill(
    description="Efface les extraits sonores gardés autour des éveils.",
    examples=["efface tes enregistrements d'éveil", "vide ton corpus d'éveil"],
    confirm="Voulez-vous vraiment que j'efface les extraits sonores d'éveil ?",
)
def oublier_le_corpus() -> str:
    """Supprime tous les extraits, vrais positifs comme faux."""
    nombre = corpus().tout_oublier()
    return f"J'ai effacé {nombre} extrait(s) sonore(s)."

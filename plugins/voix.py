"""Apprendre une voix, et s'en servir.

Ce qu'il montre :

* une compétence qui agit sur **l'audio du tour en cours**, pas sur du texte.
  Un plugin ne voit jamais le son ; c'est le pipeline qui dépose la dernière
  capture, et le plugin la reprend. Le contrat reste étroit ;
* de l'introspection honnête : « qu'est-ce que tu sais de ma voix » doit
  répondre avec des chiffres, pas avec une impression. Une reconnaissance
  qu'on ne peut pas inspecter est une reconnaissance à laquelle on ne peut
  pas faire confiance ;
* un ``confirm=`` sur l'oubli, parce que réenrôler demande de reparler
  plusieurs fois.

L'enrôlement se fait au plus simple : vous parlez, puis vous dites « apprends
ma voix ». C'est l'énoncé que vous venez de prononcer qui sert d'échantillon —
inutile de vous faire répéter une phrase imposée.
"""

from lily.plugin import SkillRefused, get_logger, locuteur, skill


@skill(
    description="Apprend à quoi ressemble votre voix, pour ne répondre qu'à vous.",
    examples=[
        "apprends ma voix",
        "retiens ma voix",
        "c'est moi qui parle",
        "reconnais-moi",
    ],
)
def apprends_ma_voix() -> dict:
    """Ajoute ce que vous venez de dire aux échantillons de votre voix.

    Répétez-le quelques fois, à des moments différents de la journée : c'est
    la variété qui fait une bonne référence, pas la longueur.
    """
    voix = locuteur()
    if not voix.actif:
        raise SkillRefused(
            "La reconnaissance de la voix est éteinte. Pour l'allumer : "
            "[voix] actif = true dans la configuration."
        )
    if not voix.disponible:
        raise SkillRefused(
            "Aucun moteur de signature vocale n'est disponible. La chaîne audio "
            'l\'apporte : pip install -e ".[audio]"'
        )

    verdict = voix.enroler()
    if not verdict:
        raise SkillRefused(verdict.raison)
    get_logger().info("Voix enrôlée : %s", verdict.raison)
    etat = voix.etat()
    return {"speak": verdict.raison.capitalize() + ".", "display": etat.decrire()}


@skill(
    description="Dit ce que Lily a appris de votre voix et si elle vous reconnaît.",
    examples=[
        "est-ce que tu reconnais ma voix",
        "qu'est-ce que tu sais de ma voix",
        "où en est ma voix",
    ],
)
def ma_voix() -> dict:
    """L'état de la reconnaissance, en clair."""
    voix = locuteur()
    etat = voix.etat()

    if not etat.actif:
        parle = (
            "Je ne cherche pas à reconnaître les voix. Pour que je m'y mette, "
            "allumez la section voix dans la configuration."
        )
    elif not voix.disponible:
        parle = "La reconnaissance est allumée, mais aucun moteur n'est disponible."
    elif not etat.enrole:
        manque = max(0, voix.echantillons_min - etat.echantillons)
        parle = (
            f"Pas encore. Dites-moi « apprends ma voix » encore {manque} fois "
            "et je saurai vous reconnaître."
        )
    else:
        parle = (
            f"Oui. J'ai {etat.echantillons} échantillons de votre voix, "
            f"et je n'ouvre la porte qu'au-dessus de {etat.seuil:.2f}."
        )
    return {"speak": parle, "display": etat.decrire()}


@skill(
    description="Efface tout ce que Lily a appris de votre voix.",
    examples=["oublie ma voix", "efface ce que tu sais de ma voix"],
    confirm="Voulez-vous vraiment que j'oublie votre voix ? Il faudra tout réapprendre.",
)
def oublie_ma_voix() -> str:
    """Remet la reconnaissance à zéro.

    Utile après un changement de micro ou de pièce : la signature porte le
    décor autant que le timbre, et une référence prise ailleurs vaut moins que
    pas de référence du tout.
    """
    nombre = locuteur().oublier()
    if not nombre:
        return "Je n'avais rien appris de votre voix."
    return (
        f"C'est oublié, {nombre} échantillon{'s' if nombre > 1 else ''} effacé"
        f"{'s' if nombre > 1 else ''}. Je répondrai de nouveau à qui m'appelle."
    )

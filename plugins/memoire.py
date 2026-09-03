"""Mémoire : ce que Lily garde d'une fois sur l'autre.

Ce qu'il montre :

* ``memoire()`` — l'accès au magasin persistant que le cœur prête aux
  plugins, au même titre que l'atelier ou le modèle ;
* deux horizons bien distincts : les **faits durables**, qui entrent dans le
  persona à chaque tour et survivent à tout, et l'**historique** des
  conversations, consultable et reprenable ;
* ``confirm=`` sur l'oubli, parce qu'effacer un souvenir sur une phrase mal
  transcrite serait particulièrement désagréable.

Tout est dans un fichier SQLite sur votre machine. Rien n'en sort.
"""

from lily.plugin import conversation, get_logger, memoire, skill

CONFIG_DEFAULTS = {
    "conversations_listees": 5,
    "extraits_par_recherche": 4,
}


@skill(
    description="Retient durablement une information sur l'utilisateur.",
    examples=[
        "retiens que je m'appelle Tom",
        "souviens-toi que mon dépôt est dans projets",
        "note dans ta mémoire que je préfère les réponses courtes",
    ],
)
def retenir(fait: str) -> str:
    """Enregistre un fait durable, réinjecté dans le persona à chaque tour.

    Args:
        fait: L'information à retenir, formulée comme une phrase.
    """
    fait = fait.strip()
    if not fait:
        return "Il me faut quelque chose à retenir."
    if memoire().retenir(fait):
        return "C'est retenu."
    return "Je le savais déjà."


@skill(
    description="Dit ce que Lily sait de son utilisateur.",
    examples=["que sais-tu de moi", "qu'est-ce que tu as retenu", "ta mémoire"],
)
def ce_que_tu_sais() -> dict:
    """Liste les faits durables enregistrés."""
    faits = memoire().faits()
    if not faits:
        return {"speak": "Je ne sais rien de particulier sur vous.", "display": "0 fait"}
    parle = " ".join(fait.contenu.rstrip(".") + "." for fait in reversed(faits[:8]))
    return {
        "speak": parle,
        "display": "\n".join(f"{fait.horodatage} — {fait.contenu}" for fait in faits),
    }


@skill(
    description="Oublie une information retenue.",
    examples=["oublie ce que tu sais sur mon adresse", "efface de ta mémoire mon prénom"],
    confirm="Voulez-vous vraiment que j'oublie cela ?",
)
def oublier(sujet: str) -> str:
    """Supprime les faits contenant un mot.

    Args:
        sujet: Le mot ou la phrase à retirer de la mémoire.
    """
    nombre = memoire().oublier(sujet)
    if not nombre:
        return f"Je n'avais rien retenu sur « {sujet} »."
    return f"{nombre} chose{'s' if nombre > 1 else ''} oubliée{'s' if nombre > 1 else ''}."


@skill(
    description="Liste les conversations passées.",
    examples=["quelles sont nos dernières conversations", "mes conversations", "l'historique"],
)
def mes_conversations() -> dict:
    """Les dernières sessions, avec leur date et leur sujet."""
    from lily.plugin import get_config

    sessions = memoire().sessions(limite=int(get_config("conversations_listees", 5)))
    if not sessions:
        return {"speak": "Nous n'avons pas encore d'historique.", "display": "0 conversation"}
    morceaux = [
        f"{session.titre or 'sans titre'}, {session.decrire().split(' — ')[1]}"
        for session in sessions[:3]
    ]
    parle = "Nos dernières conversations : " + " ; ".join(morceaux) + "."
    return {
        "speak": parle,
        "display": "\n".join(session.decrire() for session in sessions),
    }


@skill(
    description="Retrouve un passage d'une conversation passée.",
    examples=[
        "de quoi avons-nous parlé à propos du backtest",
        "retrouve ce qu'on a dit sur le pipeline",
        "cherche dans nos conversations",
    ],
)
def retrouver(sujet: str) -> dict:
    """Cherche dans tout l'historique des conversations.

    Args:
        sujet: Le mot ou la phrase à retrouver.
    """
    from lily.plugin import get_config

    extraits = memoire().chercher(sujet, limite=int(get_config("extraits_par_recherche", 4)))
    if not extraits:
        return {
            "speak": f"Je ne retrouve rien sur « {sujet} ».",
            "display": f"0 résultat pour {sujet!r}",
        }
    premier = extraits[0]
    parle = (
        f"Dans la conversation numéro {premier.session_id} : {premier.contenu[:180]}"
    )
    return {
        "speak": parle,
        "display": "\n".join(
            f"#{e.session_id} {e.horodatage} [{e.role}] {e.contenu[:120]}" for e in extraits
        ),
    }


@skill(
    description="Reprend une conversation passée là où elle s'était arrêtée.",
    examples=[
        "reprends notre conversation d'hier",
        "reprends la conversation numéro trois",
        "on reprend où on en était",
    ],
)
def reprendre_conversation(numero: int = 0) -> str:
    """Recharge une conversation dans le fil courant.

    Args:
        numero: Le numéro de la conversation. Zéro reprend la dernière.
    """
    magasin = memoire()
    cible = magasin.session(int(numero)) if numero else magasin.derniere_session()
    if cible is None:
        return "Je ne trouve pas cette conversation."

    nombre = conversation().reprendre(cible.id)
    get_logger().info("Conversation #%s reprise (%d messages).", cible.id, nombre)
    quand = cible.decrire().split(" — ")[1]
    liaison = "d'" if quand[:1].lower() in "aeiouyhé" else "de "
    return f"Nous reprenons la conversation {liaison}{quand} : {cible.titre or 'sans titre'}."

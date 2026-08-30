"""Interroger ce que Capucine a lu — documents indexés et conversations passées.

C'est la RAG (*retrieval-augmented generation*), et le mot savant recouvre une
idée simple : au lieu de demander au modèle ce qu'il croit savoir, on lui
montre d'abord les passages pertinents de **vos** documents, et on lui demande
de répondre à partir de ça.

Deux conséquences qui comptent :

* elle répond sur des choses qu'aucun modèle n'a jamais vues — vos rapports,
  vos notes, ce que vous lui avez dit la semaine dernière ;
* elle peut dire **d'où** vient la réponse, ce qui permet de la vérifier.

Sans vectoriseur installé, tout continue de marcher en plein texte : moins
fin, mais jamais absent. C'est dit à voix haute plutôt que caché.
"""

from capucine.plugin import (
    SkillRefused,
    connaissances,
    demander_au_modele,
    get_config,
    get_logger,
    skill,
)

CONFIG_DEFAULTS = {
    "passages_montres": 5,
    "caracteres_de_contexte": 4000,
    "documents_listes": 20,
}


@skill(
    description="Répond à une question à partir des documents indexés et des conversations passées.",
    examples=[
        "que sais-tu sur le backtest options",
        "d'après mes documents, combien on a perdu au premier trimestre",
        "cherche dans ce que tu as lu ce qui parle de la volatilité",
        "qu'est-ce que mes rapports disent du risque de crédit",
    ],
    timeout=180.0,
)
def que_sais_tu_sur(sujet: str) -> dict:
    """Retrouve les passages pertinents, puis répond en s'appuyant dessus.

    Args:
        sujet: La question, ou le sujet à chercher dans ce qu'elle a lu.
    """
    index = connaissances()
    passages = index.chercher(sujet, limite=int(get_config("passages_montres", 5)))
    if not passages:
        return {
            "speak": "Je ne trouve rien là-dessus dans ce que j'ai lu. "
                     "Indexez le document avec « indexe ce rapport ».",
            "display": f"Aucun passage pour « {sujet} » "
                       f"({index.statistiques()['fragments']} fragments indexés)",
        }

    maximum = int(get_config("caracteres_de_contexte", 4000))
    contexte, total = [], 0
    for passage in passages:
        bloc = passage.citer()
        if total + len(bloc) > maximum:
            break
        contexte.append(bloc)
        total += len(bloc)

    reponse = demander_au_modele(
        "Voici des extraits de mes documents, chacun précédé de sa provenance "
        f"entre crochets :\n\n{chr(10).join(contexte)}\n\n"
        f"Question : {sujet}",
        system="Tu réponds UNIQUEMENT à partir des extraits fournis, en français, "
               "en deux ou trois phrases. Nomme la provenance de ce que tu affirmes. "
               "Si les extraits ne permettent pas de répondre, dis-le franchement "
               "au lieu d'inventer.",
        max_tokens=400,
    ).strip()

    sources = "\n".join(
        f"· {passage.provenance}" + (f"  (proximité {passage.score:.2f})" if passage.score else "")
        for passage in passages
    )
    get_logger().info("Question sur les connaissances : %s (%d passages)", sujet, len(passages))
    return {
        "speak": reponse or "Je n'arrive pas à formuler de réponse à partir de ces extraits.",
        "display": f"{reponse}\n\nSources :\n{sources}",
    }


@skill(
    description="Montre les passages bruts trouvés dans les documents, sans passer par le modèle.",
    examples=[
        "montre-moi les passages sur la volatilité",
        "trouve les extraits qui parlent du pipeline",
        "cherche le passage exact",
    ],
    timeout=60.0,
)
def passages_sur(sujet: str, nombre: int = 3) -> dict:
    """Les extraits eux-mêmes, tels qu'ils sont écrits dans vos fichiers.

    Utile quand la reformulation du modèle n'est pas souhaitable : ici, rien
    n'est réécrit.

    Args:
        sujet: Ce qu'il faut chercher.
        nombre: Combien de passages montrer.
    """
    passages = connaissances().chercher(sujet, limite=max(1, int(nombre)))
    if not passages:
        raise SkillRefused(f"Rien d'indexé ne parle de « {sujet} ».")
    return {
        "speak": f"J'ai trouvé {len(passages)} passage"
                 f"{'s' if len(passages) > 1 else ''}. Le premier vient de "
                 f"{passages[0].provenance}.",
        "display": "\n\n".join(passage.citer() for passage in passages),
    }


@skill(
    description="Dit ce que Capucine a indexé : combien de documents, de fragments, et lesquels.",
    examples=[
        "qu'est-ce que tu as indexé",
        "quels documents tu connais",
        "état de tes connaissances",
    ],
)
def mes_connaissances() -> dict:
    """Le contenu de l'index, et par quel moyen il est interrogé."""
    index = connaissances()
    stats = index.statistiques()
    if not stats["fragments"]:
        return {
            "speak": "Je n'ai encore rien indexé. Dites-moi « indexe ce document ».",
            "display": "index vide",
        }

    moyen = (
        f"recherche par le sens ({stats['modele']})" if stats["vectoriel"]
        else "recherche en plein texte — aucun vectoriseur installé, "
             "faites « ollama pull nomic-embed-text » pour la recherche par le sens"
    )
    lignes = [
        f"{stats['fragments']} fragments · {stats['documents']} document(s) · "
        f"{stats['tours']} tour(s) de conversation",
        moyen,
        "",
    ]
    lignes += [
        f"{reference}  ({nombre} fragments)"
        for reference, nombre in index.references(
            "document", int(get_config("documents_listes", 20))
        )
    ]
    return {
        "speak": f"J'ai indexé {stats['documents']} document"
                 f"{'s' if stats['documents'] > 1 else ''} et "
                 f"{stats['tours']} tour{'s' if stats['tours'] > 1 else ''} de conversation.",
        "display": "\n".join(lignes),
    }


@skill(
    description="Retire un document de l'index, ou vide tout ce que Capucine a indexé.",
    examples=[
        "oublie ce document",
        "retire ce rapport de tes connaissances",
        "vide ton index",
    ],
    confirm="Voulez-vous vraiment que j'oublie ce que j'ai indexé ?",
)
def oublier_ce_que_tu_as_lu(document: str = "") -> str:
    """Efface l'index d'un document, ou l'index entier.

    Args:
        document: Le chemin du document à retirer. Vide, tout est effacé.
    """
    index = connaissances()
    if not document.strip():
        stats = index.statistiques()
        index.tout_oublier()
        return (
            f"J'ai vidé mon index : {stats['fragments']} fragment(s) effacé(s). "
            "Les fichiers eux-mêmes n'ont pas bougé."
        )
    # On accepte le chemin tel qu'il a été indexé, ou juste un nom de fichier.
    references = [
        reference for reference, _ in index.references("document", 500)
        if document.strip() in reference
    ]
    if not references:
        raise SkillRefused(f"Je n'ai rien d'indexé qui corresponde à « {document} ».")
    total = sum(index.oublier(reference) for reference in references)
    return (
        f"Oublié : {len(references)} document(s), {total} fragment(s). "
        "Les fichiers eux-mêmes n'ont pas bougé."
    )

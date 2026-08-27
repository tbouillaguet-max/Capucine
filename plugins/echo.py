"""Plugin de démonstration : le plus petit contrat possible.

Un fichier déposé dans ``plugins/``, un import, un décorateur. Rien d'autre :
pas d'enregistrement manuel, pas de manifeste, aucune ligne à ajouter dans le
cœur.
"""

from capucine.plugin import skill


@skill(
    description="Répète mot pour mot ce que l'utilisateur demande de répéter.",
    examples=[
        "répète après moi bonjour",
        "redis bonsoir",
        "répète que le café est prêt",
    ],
)
def repete(texte: str, fois: int = 1) -> str:
    """Répète un texte, éventuellement plusieurs fois.

    Le corps de cette docstring part dans le contexte du LLM : c'est ici qu'on
    précise ce qu'un nom de fonction ne dit pas.

    Args:
        texte: Ce qu'il faut répéter, tel quel.
        fois: Combien de fois le répéter, au plus cinq.
    """
    fois = max(1, min(int(fois), 5))
    return " ".join([texte] * fois)

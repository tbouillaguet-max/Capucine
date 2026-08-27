"""Plugin de démonstration : configuration, hooks, et retour dissocié.

Il montre trois éléments du contrat que ``echo.py`` n'utilise pas :

* ``CONFIG_DEFAULTS`` — des réglages surchargeables depuis ``[plugins.calcul]``
  du fichier de configuration, lus avec ``get_config()`` ;
* ``on_load()`` / ``on_unload()`` — les points d'accroche du cycle de vie ;
* un retour ``{"speak": …, "display": …}`` — ce qui est *dit* diffère de ce qui
  est *journalisé*, parce qu'une expression arithmétique se lit mal à voix haute.
"""

import ast
import operator

from capucine.plugin import get_config, get_logger, skill

CONFIG_DEFAULTS = {
    "decimales": 2,
    "expression_max": 120,
}

_OPERATEURS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def on_load() -> None:
    """Appelé une fois le module chargé, avant le premier appel."""
    get_logger().info("Calculatrice prête (%d décimales).", get_config("decimales", 2))


def on_unload() -> None:
    """Appelé avant le déchargement, pour libérer une ressource."""
    get_logger().debug("Calculatrice déchargée.")


def _evalue(noeud: ast.AST) -> float:
    """Évalue un arbre arithmétique. Jamais ``eval`` : un plugin ne doit pas
    ouvrir une porte que le cœur a pris soin de fermer."""
    if isinstance(noeud, ast.Expression):
        return _evalue(noeud.body)
    if isinstance(noeud, ast.Constant):
        if isinstance(noeud.value, bool) or not isinstance(noeud.value, (int, float)):
            raise ValueError("seuls les nombres sont acceptés")
        return noeud.value
    if isinstance(noeud, ast.BinOp) and type(noeud.op) in _OPERATEURS:
        return _OPERATEURS[type(noeud.op)](_evalue(noeud.left), _evalue(noeud.right))
    if isinstance(noeud, ast.UnaryOp) and type(noeud.op) in _OPERATEURS:
        return _OPERATEURS[type(noeud.op)](_evalue(noeud.operand))
    raise ValueError("expression non autorisée")


@skill(
    description="Calcule une expression arithmétique simple.",
    examples=[
        "combien font 12 fois 8",
        "calcule 145 divisé par 5",
        "quelle est la racine de 2 plus 3",
    ],
)
def calculer(expression: str) -> dict:
    """Évalue une expression arithmétique et en donne le résultat.

    Args:
        expression: L'expression à calculer, en notation mathématique
            (« 12 * 8 », « (3 + 4) / 2 »).
    """
    decimales = int(get_config("decimales", 2))
    longueur_max = int(get_config("expression_max", 120))

    nettoyee = (
        expression.lower()
        .replace("×", "*").replace("÷", "/").replace("^", "**")
        .replace(",", ".").replace("fois", "*").replace("plus", "+")
        .replace("moins", "-").replace("divisé par", "/").replace("divise par", "/")
        .strip()
    )
    if len(nettoyee) > longueur_max:
        return {"speak": "Cette expression est trop longue pour moi.",
                "display": f"expression rejetée ({len(nettoyee)} caractères)"}

    try:
        valeur = _evalue(ast.parse(nettoyee, mode="eval"))
    except ZeroDivisionError:
        return {"speak": "Une division par zéro, ce n'est pas possible.",
                "display": f"{expression} -> division par zéro"}
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        return {"speak": "Je n'ai pas compris cette expression.",
                "display": f"{expression} -> {exc}"}

    arrondi = round(valeur, decimales)
    if isinstance(arrondi, float) and arrondi.is_integer():
        arrondi = int(arrondi)
    return {"speak": f"Ça fait {arrondi}.", "display": f"{expression} = {arrondi}"}

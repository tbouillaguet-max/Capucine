"""Écrire et exécuter du Python.

Ce qu'il montre :

* ``demander_au_modele()`` — un plugin peut solliciter le modèle local pour
  produire du texte, ici du code. C'est une complétion simple : pas de
  routage, donc pas de récursion possible ;
* un cycle **proposer puis enregistrer** en deux compétences. Le code généré
  est d'abord montré, et n'atterrit sur le disque qu'après un accord explicite.
  Un modèle 7B écrit du Python approximatif : on ne l'écrit jamais à l'aveugle ;
* de l'exécution en sous-processus, avec délai, répertoire courant contraint
  à l'atelier, et jamais de shell.

Tout ce qui touche au disque ou lance du code demande confirmation.
"""

import shlex
import subprocess
import sys

from capucine.plugin import (
    SkillRefused,
    atelier,
    demander_au_modele,
    get_config,
    get_logger,
    skill,
)

CONFIG_DEFAULTS = {
    "delai_s": 60.0,
    "sortie_max_lignes": 40,
    "interpreteur": "",          # vide = le même Python que Capucine
    "max_tokens_code": 900,
}

CONSIGNE = (
    "Tu écris du Python 3.11 clair et court. Réponds UNIQUEMENT par du code, "
    "sans texte autour et sans balises Markdown. Le code doit être complet et "
    "exécutable tel quel, avec une docstring en tête et des noms explicites."
)

# La dernière proposition du modèle, en attente d'un accord pour être écrite.
_PROPOSITION: dict = {"code": "", "description": ""}


def _interpreteur() -> str:
    return str(get_config("interpreteur", "")) or sys.executable


def _nettoyer(code: str) -> str:
    """Retire les balises Markdown que les modèles ajoutent malgré la consigne."""
    code = code.strip()
    if code.startswith("```"):
        lignes = code.splitlines()
        lignes = lignes[1:]
        if lignes and lignes[-1].strip().startswith("```"):
            lignes = lignes[:-1]
        code = "\n".join(lignes)
    return code.strip() + "\n"


def _executer(arguments: list[str], delai: float) -> dict:
    espace = atelier()
    try:
        resultat = subprocess.run(  # noqa: S603 - jamais de shell, arguments listés
            arguments,
            cwd=str(espace.racines[0]),
            capture_output=True,
            text=True,
            timeout=delai,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise SkillRefused(
            f"Le programme tournait encore après {delai:.0f} secondes, je l'ai arrêté."
        ) from None
    except OSError as exc:
        raise SkillRefused(f"Impossible de lancer l'interpréteur : {exc}") from None

    maximum = int(get_config("sortie_max_lignes", 40))
    sortie = (resultat.stdout or "").strip().splitlines()
    erreurs = (resultat.stderr or "").strip().splitlines()
    return {
        "code_retour": resultat.returncode,
        "sortie": "\n".join(sortie[-maximum:]),
        "erreurs": "\n".join(erreurs[-maximum:]),
    }


def _resumer(resultat: dict, quoi: str) -> dict:
    if resultat["code_retour"] == 0:
        premiere = resultat["sortie"].splitlines()[:3]
        parle = f"{quoi} a fonctionné. " + " ".join(premiere) if premiere else f"{quoi} a fonctionné, sans rien afficher."
    else:
        derniere = resultat["erreurs"].splitlines()[-1:] or ["sans message"]
        parle = f"{quoi} a échoué : {derniere[0]}"
    journal = f"code de retour {resultat['code_retour']}"
    if resultat["sortie"]:
        journal += f"\n--- sortie ---\n{resultat['sortie']}"
    if resultat["erreurs"]:
        journal += f"\n--- erreurs ---\n{resultat['erreurs']}"
    return {"speak": parle[:500], "display": journal}


@skill(
    description="Exécute un bout de code Python et donne son résultat.",
    examples=["exécute ce code python", "calcule ça en python", "lance ce bout de code"],
    confirm="Voulez-vous vraiment que j'exécute ce code ?",
    timeout=120.0,
)
def executer_python(code: str) -> dict:
    """Lance du Python dans un sous-processus, depuis l'atelier.

    Args:
        code: Le code à exécuter.
    """
    code = _nettoyer(code)
    get_logger().info("Exécution de %d caractères de Python.", len(code))
    resultat = _executer([_interpreteur(), "-c", code], float(get_config("delai_s", 60.0)))
    return _resumer(resultat, "Le code")


@skill(
    description="Lance un script Python du projet.",
    examples=["lance le script de backtest", "exécute zéro cinq calcul multiples", "fais tourner ce script"],
    confirm="Voulez-vous vraiment lancer ce script ?",
    timeout=600.0,
)
def lancer_script(chemin: str, arguments: str = "") -> dict:
    """Exécute un fichier Python de l'atelier.

    Args:
        chemin: Le script à lancer, relatif à l'atelier.
        arguments: Les options de ligne de commande, séparées par des espaces.
    """
    cible = atelier().resoudre(chemin, doit_exister=True)
    if cible.suffix != ".py":
        raise SkillRefused(f"« {cible.name} » n'est pas un script Python.")

    # shlex plutôt que split() : « --as-of-date "2025-06-30" » doit rester
    # un seul argument, guillemets compris.
    commande = [_interpreteur(), str(cible), *shlex.split(arguments)]
    get_logger().info("Lancement : %s", " ".join(commande))
    resultat = _executer(commande, float(get_config("delai_s", 60.0)) * 5)
    return _resumer(resultat, cible.name)


@skill(
    description="Écrit du code Python à partir d'une description, et le montre.",
    examples=[
        "écris-moi un script qui trie un fichier csv",
        "code une fonction qui calcule une moyenne mobile",
        "propose-moi du python pour lire ce json",
    ],
    timeout=120.0,
)
def ecrire_du_code(description: str) -> dict:
    """Demande du code au modèle local, sans l'écrire nulle part.

    Le code reste en attente : ``enregistrer_le_code`` le pose sur le disque
    après votre accord. On ne lui fait jamais confiance à l'aveugle.

    Args:
        description: Ce que le code doit faire.
    """
    description = description.strip()
    if not description:
        return {"speak": "Que doit faire ce code ?", "display": "description vide"}

    code = _nettoyer(demander_au_modele(
        description, system=CONSIGNE,
        max_tokens=int(get_config("max_tokens_code", 900)), temperature=0.1,
    ))
    if not code.strip():
        return {"speak": "Le modèle n'a rien produit.", "display": "réponse vide"}

    _PROPOSITION["code"] = code
    _PROPOSITION["description"] = description
    lignes = code.splitlines()
    return {
        "speak": (
            f"J'ai écrit {len(lignes)} lignes. Relisez-les, puis dites-moi où "
            "les enregistrer."
        ),
        "display": code,
    }


@skill(
    description="Enregistre le dernier code proposé dans un fichier.",
    examples=["enregistre ce code", "écris-le dans un fichier", "sauvegarde la proposition"],
    confirm="Voulez-vous vraiment enregistrer ce code ? Je garde une sauvegarde de l'existant.",
)
def enregistrer_le_code(chemin: str) -> str:
    """Écrit la dernière proposition dans l'atelier.

    Args:
        chemin: Où enregistrer le fichier, relatif à l'atelier.
    """
    if not _PROPOSITION["code"]:
        raise SkillRefused("Je n'ai aucun code en attente. Demandez-m'en un d'abord.")
    cible = atelier().ecrire(chemin, _PROPOSITION["code"])
    lignes = len(_PROPOSITION["code"].splitlines())
    _PROPOSITION["code"] = ""
    return f"{lignes} lignes écrites dans {cible.name}."


@skill(
    description="Explique ce que fait un fichier de code.",
    examples=["explique-moi ce fichier", "que fait ce script", "résume ce code"],
    timeout=120.0,
)
def expliquer_le_code(chemin: str) -> dict:
    """Fait résumer un fichier par le modèle local.

    Args:
        chemin: Le fichier à expliquer, relatif à l'atelier.
    """
    contenu = atelier().lire(chemin)
    if len(contenu) > 12000:
        contenu = contenu[:12000] + "\n# … fichier tronqué"
    reponse = demander_au_modele(
        f"Explique en trois phrases, en français, ce que fait ce fichier :\n\n{contenu}",
        system="Tu expliques du code brièvement et précisément. Pas de liste, pas de titre.",
        max_tokens=300,
    ).strip()
    return {"speak": reponse or "Je n'arrive pas à le résumer.", "display": reponse}

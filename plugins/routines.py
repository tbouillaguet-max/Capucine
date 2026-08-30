"""Apprendre une routine en la montrant, pas en la programmant.

Vous faites trois choses. Vous dites « retiens ça, c'est ma routine du
matin ». Capucine **écrit un plugin** dans ``plugins/``, le rechargement à
chaud le voit passer, et la compétence existe — sans redémarrage, sans que
personne n'ait touché au cœur.

C'est la contrainte numéro un du projet retournée comme un gant : ajouter une
capacité, c'est déposer un fichier dans ``plugins/``. Y compris quand c'est
Capucine qui le dépose.

Deux choses qu'elle ne fait **pas**, délibérément :

* **Le modèle n'écrit pas ce fichier.** Le code sort d'un gabarit fixe ; du
  journal ne viennent que des noms de compétences existantes et des arguments
  passés par ``json.dumps``. Un plugin généré ne peut donc contenir que des
  appels à ce que vous venez de faire — pas du code inventé.
* **Elle ne contourne aucune confirmation.** Une étape déclarée ``confirm=``
  reste bloquante à l'exécution de la routine ; elle ne peut pas effacer vos
  notes en passant.

Le fichier produit est du Python lisible, dans votre dossier ``plugins/``.
Ouvrez-le, modifiez-le, supprimez-le : c'est un fichier comme un autre.
"""

import ast
import json
import os
import re
from datetime import date
from pathlib import Path

from capucine.plugin import (
    SkillRefused,
    appeler_competence,
    dossier_des_plugins,
    get_config,
    get_logger,
    journal,
    skill,
)

CONFIG_DEFAULTS = {
    "etapes_par_defaut": 3,
    "etapes_max": 8,
}

PREFIXE = "routine_"

GABARIT = '''"""Routine « {titre} », écrite par Capucine le {date}.

Apprise en la montrant : les {nombre} gestes ci-dessous sont ceux qui venaient
d'être faits quand vous avez dit « retiens cette routine ».

C'est un fichier comme un autre. Modifiez l'ordre, changez un argument,
ajoutez une étape — le rechargement à chaud prendra la nouvelle version.
Supprimez-le et la routine disparaît.
"""

from capucine.plugin import appeler_competence, skill

ETAPES = {etapes}


@skill(
    description="{description}",
    examples={exemples},
    timeout={timeout},
)
def {fonction}() -> dict:
    """{docstring}"""
    dits, montres = [], []
    for competence, arguments in ETAPES:
        resultat = appeler_competence(competence, arguments)
        if resultat.needs_confirmation:
            # Une étape qui demande confirmation arrête la routine : on ne
            # contourne pas une garde en l'enrobant dans un enchaînement.
            dits.append(resultat.speak)
            montres.append(f"{{competence}} : confirmation demandée, routine interrompue")
            break
        dits.append(resultat.speak or "")
        montres.append(f"{{competence}} : {{resultat.display or resultat.speak}}")
    return {{"speak": " ".join(part for part in dits if part),
            "display": "\\n".join(montres)}}
'''


# Ce qui précède le nom quand on le dicte. Sans ce ménage, « retiens cette
# routine, elle s'appelle mon matin » deviendrait une compétence nommée
# « retiens_cette_routine_elle_s_appelle_mon » — et personne ne la rappellerait
# jamais par ce nom-là.
AMORCES = (
    "retiens cette routine", "retiens la routine", "retiens cet enchainement",
    "retiens ca", "retiens", "garde cet enchainement", "garde ca", "garde",
    "fais en une routine", "fais en", "enregistre cette routine", "enregistre",
    "comme routine", "en routine", "sous le nom de", "sous le nom",
    "elle s appelle", "il s appelle", "ca s appelle", "appelle la", "appelle ca",
    "c est ma routine", "c est la routine", "c est", "ma routine", "la routine",
    "routine", "et", "de", "du", "des", "le", "la", "les", "mon", "ma", "mes",
)
MOTS_MAX = 4
_JETONS = re.compile(r"[^\W_]+", re.UNICODE)


def _nettoyer_le_nom(dicte: str) -> str:
    """Extrait le nom utile de ce qui a été dicté.

    « Retiens cette routine, elle s'appelle mon matin » → « matin ». Le nom
    sert à rappeler la routine à la voix : quatre mots au plus, sans les
    formules qui introduisent la demande, et avec ses accents.

    Les articles ne sont retirés qu'en tête, jamais au milieu : « le tour du
    matin » garde son « du ».
    """
    # Deux découpages du MÊME texte, mot pour mot : l'un sert à reconnaître
    # les amorces, l'autre à rendre le nom tel qu'il a été dit.
    originaux = _JETONS.findall(dicte)
    normalises = [_sans_accents(mot).lower() for mot in originaux]

    change = True
    while change and normalises:
        change = False
        for amorce in AMORCES:
            morceaux = amorce.split()
            if len(normalises) > len(morceaux) and normalises[: len(morceaux)] == morceaux:
                del normalises[: len(morceaux)]
                del originaux[: len(morceaux)]
                change = True
                break
    return " ".join(originaux[:MOTS_MAX])


def _nom_de_fichier(nom: str) -> str:
    """Un nom de fichier sûr, dérivé de ce que vous avez dicté."""
    propre = "".join(
        caractere if caractere.isalnum() else "_"
        for caractere in _sans_accents(nom.strip().lower())
    ).strip("_")
    propre = "_".join(morceau for morceau in propre.split("_") if morceau)
    if not propre or not propre[0].isalpha():
        propre = f"r_{propre}" if propre else ""
    if not propre:
        raise SkillRefused("Ce nom de routine ne donne aucun nom de fichier utilisable.")
    return propre[:40]


def _sans_accents(texte: str) -> str:
    import unicodedata

    return "".join(
        caractere for caractere in unicodedata.normalize("NFKD", texte)
        if not unicodedata.combining(caractere)
    )


def _etapes_en_python(appels) -> str:
    """La liste des étapes, en Python lisible et sûr.

    Les noms viennent du registre, les arguments passent par ``json.dumps`` :
    rien de ce qui est écrit ici ne peut être autre chose qu'une chaîne ou
    une valeur littérale.
    """
    lignes = [
        f"    ({json.dumps(appel.competence, ensure_ascii=False)}, "
        f"{json.dumps(appel.arguments, ensure_ascii=False)}),"
        for appel in appels
    ]
    return "[\n" + "\n".join(lignes) + "\n]"


def _fichiers_de_routines() -> list[Path]:
    return sorted(dossier_des_plugins().glob(f"{PREFIXE}*.py"))


@skill(
    description="Retient les dernières actions faites et en fait une routine réutilisable.",
    examples=[
        "retiens cette routine",
        "retiens ça, c'est ma routine du matin",
        "fais-en une routine",
        "garde cet enchaînement",
    ],
)
def retenir_cette_routine(nom: str, etapes: int = 0) -> dict:
    """Écrit un plugin qui rejoue les dernières compétences appelées.

    Args:
        nom: Le nom de la routine, par exemple « mon matin ».
        etapes: Combien de gestes récents reprendre. Zéro prend la valeur configurée.
    """
    nombre = int(etapes) or int(get_config("etapes_par_defaut", 3))
    nombre = max(1, min(nombre, int(get_config("etapes_max", 8))))
    appels = [
        appel for appel in journal().recents(nombre)
        # Une routine qui se retient elle-même serait un joli serpent qui se
        # mord la queue, et un plugin inutilisable.
        if appel.competence not in ("retenir_cette_routine", "mes_routines")
    ]
    if not appels:
        raise SkillRefused(
            "Je n'ai rien fait de récent à retenir. Demandez-moi d'abord "
            "deux ou trois choses, puis dites « retiens cette routine »."
        )

    titre = _nettoyer_le_nom(nom) or nom.strip()
    if not titre:
        raise SkillRefused("Donnez-lui un nom : « retiens cette routine, mon matin ».")
    fonction = _nom_de_fichier(titre)
    chemin = dossier_des_plugins() / f"{PREFIXE}{fonction}.py"
    existait = chemin.exists()

    resume = ", puis ".join(appel.competence.replace("_", " ") for appel in appels)
    code = GABARIT.format(
        titre=titre,
        date=date.today().isoformat(),
        nombre=len(appels),
        etapes=_etapes_en_python(appels),
        description=f"Routine « {titre} » : {resume}.".replace('"', "'"),
        exemples=json.dumps(
            [titre, f"lance {titre}", f"fais {titre}", "ma routine"], ensure_ascii=False
        ),
        timeout=round(10.0 * len(appels) + 20.0, 1),
        fonction=fonction,
        docstring=f"Enchaîne : {resume}.".replace('"', "'"),
    )

    # Écriture atomique : le surveillant ne doit jamais voir un fichier à
    # moitié écrit. Le fichier temporaire n'a pas l'extension .py, donc il ne
    # déclenche rien ; le renommage, lui, déclenche un rechargement propre.
    temporaire = chemin.with_suffix(".routine-tmp")
    temporaire.write_text(code, encoding="utf-8")
    os.replace(temporaire, chemin)
    get_logger().info("Routine écrite : %s (%d étapes)", chemin.name, len(appels))

    return {
        "speak": f"C'est retenu. Dites « {titre} » et je referai : {resume}."
                 + (" J'ai remplacé la routine du même nom." if existait else ""),
        "display": f"{chemin}\n\n" + "\n".join(
            f"  {index}. {appel.decrire()}" for index, appel in enumerate(appels, 1)
        ) + "\n\nLe rechargement à chaud la prendra d'ici une seconde — "
            "c'est le même chemin que pour un plugin que vous déposeriez vous-même.",
    }


@skill(
    description="Dit quelles routines Capucine a apprises et ce qu'elles enchaînent.",
    examples=["quelles routines tu connais", "mes routines", "montre-moi mes enchaînements"],
)
def mes_routines() -> dict:
    """La liste des routines écrites, avec leurs étapes."""
    fichiers = _fichiers_de_routines()
    if not fichiers:
        return {
            "speak": "Je n'ai appris aucune routine. Faites deux ou trois choses, "
                     "puis dites « retiens cette routine ».",
            "display": "aucune routine",
        }
    lignes = []
    for fichier in fichiers:
        nom = fichier.stem[len(PREFIXE):]
        try:
            texte = fichier.read_text(encoding="utf-8")
            debut = texte.index("ETAPES = ") + len("ETAPES = ")
            # `literal_eval` et pas `json.loads` : les étapes sont des tuples
            # Python. Il n'évalue que des littéraux, jamais du code.
            etapes = ast.literal_eval(texte[debut : texte.index("\n\n\n@skill", debut)])
            detail = " → ".join(competence for competence, _ in etapes)
        except (OSError, ValueError, SyntaxError):
            detail = "(illisible)"
        lignes.append(f"{nom:<20} {detail}")
    return {
        "speak": f"Je connais {len(fichiers)} routine{'s' if len(fichiers) > 1 else ''} : "
                 + ", ".join(fichier.stem[len(PREFIXE):] for fichier in fichiers) + ".",
        "display": "\n".join(lignes),
    }


@skill(
    description="Supprime une routine apprise.",
    examples=["oublie la routine du matin", "supprime cette routine"],
    confirm="Voulez-vous vraiment que je supprime cette routine ?",
)
def oublier_la_routine(nom: str) -> str:
    """Efface le fichier d'une routine.

    Args:
        nom: Le nom de la routine à supprimer.
    """
    cherche = _nom_de_fichier(nom)
    for fichier in _fichiers_de_routines():
        if fichier.stem[len(PREFIXE):] == cherche:
            fichier.unlink()
            return f"La routine « {nom} » est supprimée. Elle disparaîtra d'ici une seconde."
    raise SkillRefused(f"Je ne connais pas de routine « {nom} ».")


@skill(
    description="Exécute une compétence par son nom, telle quelle.",
    examples=["exécute la compétence heure", "lance la compétence mes notes"],
)
def executer_la_competence(competence: str) -> dict:
    """Appelle une compétence sans argument, pour composer à la main.

    Args:
        competence: Le nom exact de la compétence à exécuter.
    """
    resultat = appeler_competence(competence.strip().replace(" ", "_"))
    return {
        "speak": resultat.speak,
        "display": resultat.display or resultat.speak,
    }

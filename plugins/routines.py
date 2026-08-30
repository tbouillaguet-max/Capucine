"""Trois façons d'apprendre une capacité — jamais en la programmant à l'aveugle.

Vous faites trois choses. Vous dites « retiens ça, c'est ma routine du
matin ». Capucine **écrit un plugin** dans ``plugins/``, le rechargement à
chaud le voit passer, et la compétence existe — sans redémarrage, sans que
personne n'ait touché au cœur.

C'est la contrainte numéro un du projet retournée comme un gant : ajouter une
capacité, c'est déposer un fichier dans ``plugins/``. Y compris quand c'est
Capucine qui le dépose. Ce fichier offre trois portes vers ça, avec trois
garanties différentes :

* **La montrer** (``retenir_cette_routine``) — rejoue ce qui vient d'être
  fait. Le modèle n'intervient pas : les noms viennent du journal, les
  arguments passent par ``json.dumps``.
* **La décrire** (``composer_une_capacite``) — une phrase suffit, sans avoir
  rien fait avant. Le modèle choisit des noms de compétences *déjà chargées*
  dans une liste qu'on lui donne ; un nom hors de cette liste fait échouer la
  composition entière, jamais une capacité à moitié inventée.
* **En demander une neuve** (``proposer_une_capacite`` puis
  ``activer_la_capacite_proposee``) — pour ce qu'aucune compétence existante
  ne couvre. Là, et seulement là, le modèle écrit du code. Il part dans un
  dossier à l'écart de ``plugins/`` que le surveillant ne regarde pas : rien
  ne se charge tout seul. Il faut une relecture, puis un accord explicite
  pour qu'il rejoigne ``plugins/``.

Dans les trois cas : une étape déclarée ``confirm=`` reste bloquante à
l'exécution — aucune capacité, montrée, composée ou écrite, ne peut
contourner une garde en l'enrobant. Et le fichier produit reste du Python
lisible. Ouvrez-le, modifiez-le, supprimez-le : c'est un fichier comme un
autre.
"""

import ast
import json
import os
import re
import shutil
from datetime import date, datetime
from pathlib import Path

from capucine.plugin import (
    SkillRefused,
    appeler_competence,
    competences_disponibles,
    data_dir,
    demander_au_modele,
    dossier_des_plugins,
    get_config,
    get_logger,
    journal,
    skill,
)

CONFIG_DEFAULTS = {
    "etapes_par_defaut": 3,
    "etapes_max": 8,
    "max_tokens_capacite": 900,
}

PREFIXE = "routine_"

GABARIT = '''"""Routine « {titre} », écrite par Capucine le {date}.

{origine}

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

ORIGINE_MONTREE = (
    "Apprise en la montrant : les {nombre} gestes ci-dessous sont ceux qui venaient\n"
    "d'être faits quand vous avez dit « retiens cette routine »."
)
ORIGINE_COMPOSEE = (
    "Apprise en la décrivant : les {nombre} étapes ci-dessous sont celles que le\n"
    "modèle a choisies, parmi les compétences déjà connues, pour la demande dictée."
)


# Ce qui précède le nom quand on le dicte. Sans ce ménage, « retiens cette
# routine, elle s'appelle mon matin » deviendrait une compétence nommée
# « retiens_cette_routine_elle_s_appelle_mon » — et personne ne la rappellerait
# jamais par ce nom-là.
AMORCES = (
    "retiens cette routine", "retiens la routine", "retiens cet enchainement",
    "retiens ca", "retiens", "garde cet enchainement", "garde ca", "garde",
    "fais en une routine", "fais en", "enregistre cette routine", "enregistre",
    "compose une capacite", "compose moi une capacite", "compose une routine",
    "cree une capacite", "cree moi une capacite", "cree toi une capacite",
    "cree une routine", "ecris une capacite", "ecris moi une capacite",
    "invente une capacite", "invente une competence", "pour",
    "comme routine", "en routine", "sous le nom de", "sous le nom",
    "elle s appelle", "il s appelle", "ca s appelle", "appelle la", "appelle ca",
    "c est ma routine", "c est la routine", "c est", "ma routine", "la routine",
    "routine", "capacite", "competence", "et", "de", "du", "des",
    "le", "la", "les", "mon", "ma", "mes",
)
MOTS_MAX = 4
_JETONS = re.compile(r"[^\W_]+", re.UNICODE)

# Compétences de ce fichier : jamais proposées au modèle comme brique d'une
# composition, jamais rejouées par une routine. Une capacité qui se compose
# elle-même serait un joli serpent qui se mord la queue.
NOMS_INTERNES = frozenset({
    "retenir_cette_routine", "mes_routines", "oublier_la_routine",
    "executer_la_competence", "composer_une_capacite", "proposer_une_capacite",
    "activer_la_capacite_proposee", "rejeter_la_proposition", "mes_propositions",
})

CONSIGNE_COMPOSITION = (
    "Tu composes un enchaînement de compétences déjà existantes pour Capucine, "
    "une assistante vocale francophone. Réponds UNIQUEMENT par une liste JSON, "
    "sans texte autour ni balises Markdown. Chaque élément est un objet "
    '{"competence": "<nom exact tiré de la liste fournie>", "arguments": {}}. '
    "N'utilise QUE des noms tirés de la liste fournie ; n'en invente aucun, "
    "même s'il te semble plausible. Si rien dans la liste ne convient à la "
    "demande, réponds par une liste vide []."
)

CONSIGNE_CAPACITE = (
    "Tu écris un plugin Python pour Capucine, une assistante vocale locale et "
    "francophone. Contrat strict, à respecter à la lettre :\n"
    "- Un seul import de compétences : `from capucine.plugin import skill`, et "
    "au besoin `get_config`, `atelier`, `data_dir`, `demander_au_modele` du "
    "même module — jamais d'autre dépendance interne au projet.\n"
    "- Au moins une fonction décorée par `@skill(description=\"...\", "
    "examples=[\"...\"])`, qui rend soit une chaîne, soit un dict "
    "{\"speak\": \"...\", \"display\": \"...\"}.\n"
    "- Aucun accès disque hors de `atelier()`, aucun réseau, aucun "
    "`subprocess`, `os.system`, `eval`, `exec` ni `__import__`.\n"
    "- Code complet et exécutable tel quel, Python 3.11, avec une courte "
    "docstring de module.\n"
    "Réponds UNIQUEMENT par le code du fichier, sans texte autour ni balises "
    "Markdown."
)

# Une compétence qui contiendrait un de ces jetons ne s'installe pas toute
# seule : elle sort de ce qu'un plugin de Capucine est censé toucher, et
# mérite un œil humain avant d'atterrir dans plugins/.
JETONS_RISQUES = (
    "subprocess", "os.system", "os.popen", "eval(", "exec(", "__import__",
    "socket.", "shutil.rmtree", "requests.", "urllib.", "ftplib", "smtplib",
    "ctypes", "pickle.loads",
)


def _nettoyer_le_nom(dicte: str) -> str:
    """Extrait le nom utile de ce qui a été dicté.

    « Retiens cette routine, elle s'appelle mon matin » → « matin ». Le nom
    sert à rappeler la capacité à la voix : quatre mots au plus, sans les
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
        raise SkillRefused("Ce nom ne donne aucun nom de fichier utilisable.")
    return propre[:40]


def _sans_accents(texte: str) -> str:
    import unicodedata

    return "".join(
        caractere for caractere in unicodedata.normalize("NFKD", texte)
        if not unicodedata.combining(caractere)
    )


def _etapes_en_python(etapes: list[tuple[str, dict]]) -> str:
    """La liste des étapes, en Python lisible et sûr.

    Les noms viennent du registre, les arguments passent par ``json.dumps`` :
    rien de ce qui est écrit ici ne peut être autre chose qu'une chaîne ou
    une valeur littérale.
    """
    lignes = [
        f"    ({json.dumps(competence, ensure_ascii=False)}, "
        f"{json.dumps(arguments, ensure_ascii=False)}),"
        for competence, arguments in etapes
    ]
    return "[\n" + "\n".join(lignes) + "\n]"


def _decrire_etape(competence: str, arguments: dict) -> str:
    if not arguments:
        return competence
    details = ", ".join(f"{cle}={valeur!r}" for cle, valeur in arguments.items())
    return f"{competence}({details})"


def _resumer_les_etapes(etapes: list[tuple[str, dict]]) -> str:
    return ", puis ".join(competence.replace("_", " ") for competence, _ in etapes)


def _fichiers_de_routines() -> list[Path]:
    return sorted(dossier_des_plugins().glob(f"{PREFIXE}*.py"))


def _ecrire_le_plugin_de_routine(
    titre: str, etapes: list[tuple[str, dict]], fonction: str, origine: str
) -> tuple[Path, bool]:
    """Le cœur commun aux deux façons d'apprendre sans code inventé.

    ``origine`` distingue seulement le commentaire de tête du fichier —
    montrée ou composée — jamais la façon dont il est écrit ni exécuté.
    """
    chemin = dossier_des_plugins() / f"{PREFIXE}{fonction}.py"
    existait = chemin.exists()
    resume = _resumer_les_etapes(etapes)
    code = GABARIT.format(
        titre=titre,
        date=date.today().isoformat(),
        origine=origine.format(nombre=len(etapes)),
        etapes=_etapes_en_python(etapes),
        description=f"Routine « {titre} » : {resume}.".replace('"', "'"),
        exemples=json.dumps(
            [titre, f"lance {titre}", f"fais {titre}", "ma routine"], ensure_ascii=False
        ),
        timeout=round(10.0 * len(etapes) + 20.0, 1),
        fonction=fonction,
        docstring=f"Enchaîne : {resume}.".replace('"', "'"),
    )

    # Écriture atomique : le surveillant ne doit jamais voir un fichier à
    # moitié écrit. Le fichier temporaire n'a pas l'extension .py, donc il ne
    # déclenche rien ; le renommage, lui, déclenche un rechargement propre.
    temporaire = chemin.with_suffix(".routine-tmp")
    temporaire.write_text(code, encoding="utf-8")
    os.replace(temporaire, chemin)
    return chemin, existait


def _nettoyer_le_code(code: str) -> str:
    """Retire les balises Markdown que les modèles ajoutent malgré la consigne."""
    code = code.strip()
    if code.startswith("```"):
        lignes = code.splitlines()
        lignes = lignes[1:]
        if lignes and lignes[-1].strip().startswith("```"):
            lignes = lignes[:-1]
        code = "\n".join(lignes)
    return code.strip() + "\n"


def _jetons_risques_presents(code: str) -> list[str]:
    return [jeton for jeton in JETONS_RISQUES if jeton in code]


def _dossier_des_propositions() -> Path:
    """Un dossier à l'écart de plugins/ : rien ici ne se charge tout seul."""
    dossier = data_dir() / "propositions"
    dossier.mkdir(parents=True, exist_ok=True)
    return dossier


# --- l'apprendre en la montrant ----------------------------------------------

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
        if appel.competence not in NOMS_INTERNES
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
    etapes_liste = [(appel.competence, appel.arguments) for appel in appels]

    chemin, existait = _ecrire_le_plugin_de_routine(titre, etapes_liste, fonction, ORIGINE_MONTREE)
    get_logger().info("Routine écrite : %s (%d étapes)", chemin.name, len(etapes_liste))

    resume = _resumer_les_etapes(etapes_liste)
    return {
        "speak": f"C'est retenu. Dites « {titre} » et je referai : {resume}."
                 + (" J'ai remplacé la routine du même nom." if existait else ""),
        "display": f"{chemin}\n\n" + "\n".join(
            f"  {index}. {_decrire_etape(competence, arguments)}"
            for index, (competence, arguments) in enumerate(etapes_liste, 1)
        ) + "\n\nLe rechargement à chaud la prendra d'ici une seconde — "
            "c'est le même chemin que pour un plugin que vous déposeriez vous-même.",
    }


# --- l'apprendre en la décrivant ---------------------------------------------

def _catalogue(disponibles: dict[str, str]) -> str:
    return "\n".join(f"- {nom} : {description}" for nom, description in sorted(disponibles.items()))


def _etapes_depuis_le_modele(reponse: str, disponibles: dict[str, str]) -> list[tuple[str, dict]]:
    """Traduit la réponse du modèle en étapes, ou refuse — jamais un mélange.

    Un nom hors de ``disponibles`` fait échouer toute la composition : mieux
    vaut dire clairement qu'on ne sait pas faire que d'installer une capacité
    amputée d'une étape que le modèle aurait inventée.
    """
    reponse = _nettoyer_le_code(reponse).strip()
    try:
        brut = json.loads(reponse)
    except json.JSONDecodeError:
        raise SkillRefused(
            "Le modèle n'a pas répondu par une liste exploitable ; reformulez la demande."
        ) from None
    if not isinstance(brut, list):
        raise SkillRefused("Le modèle n'a pas répondu par une liste d'étapes ; reformulez la demande.")

    etapes: list[tuple[str, dict]] = []
    for item in brut:
        if not isinstance(item, dict) or "competence" not in item:
            raise SkillRefused("Une étape proposée par le modèle est mal formée.")
        competence = str(item["competence"]).strip()
        arguments = item.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise SkillRefused(f"Les arguments de « {competence} » ne sont pas exploitables.")
        if competence not in disponibles:
            raise SkillRefused(
                f"« {competence} » n'est pas une compétence que je connais déjà : "
                "je ne compose qu'avec ce qui existe, jamais en inventant un nom."
            )
        etapes.append((competence, {str(cle): valeur for cle, valeur in arguments.items()}))
    return etapes


@skill(
    description="Compose une nouvelle capacité en enchaînant des compétences déjà connues, à partir d'une description.",
    examples=[
        "crée une capacité qui donne l'heure puis mes notes",
        "compose-moi une routine qui donne l'état du système puis l'heure",
        "crée-toi une capacité pour préparer mon café",
    ],
    timeout=30.0,
)
def composer_une_capacite(nom: str, description: str) -> dict:
    """Écrit un plugin qui enchaîne des compétences choisies depuis une phrase.

    Contrairement à ``retenir_cette_routine``, qui rejoue ce qui vient d'être
    fait, l'enchaînement n'a pas besoin d'avoir été exécuté avant : le modèle
    choisit seulement des noms parmi les compétences déjà chargées. Il
    n'écrit aucun code — le fichier sort du même gabarit fixe que pour une
    routine montrée.

    Args:
        nom: Le nom de la capacité, par exemple « bilan du matin ».
        description: Ce que la capacité doit enchaîner, en une phrase.
    """
    titre = _nettoyer_le_nom(nom) or nom.strip()
    if not titre:
        raise SkillRefused("Donnez-lui un nom : « compose une capacité, bilan du matin ».")
    description = description.strip()
    if not description:
        raise SkillRefused("Que doit enchaîner cette capacité ?")

    disponibles = {
        nom_: description_
        for nom_, description_ in competences_disponibles().items()
        if nom_ not in NOMS_INTERNES
    }
    reponse = demander_au_modele(
        f"Compétences disponibles :\n{_catalogue(disponibles)}\n\nDemande : « {description} »",
        system=CONSIGNE_COMPOSITION,
        max_tokens=400,
        temperature=0.0,
    )
    etapes = _etapes_depuis_le_modele(reponse, disponibles)
    if not etapes:
        raise SkillRefused(
            "Aucune compétence existante ne couvre cette demande. Essayez "
            "« écris une capacité qui... » pour du code neuf, à relire avant de l'installer."
        )
    maximum = int(get_config("etapes_max", 8))
    if len(etapes) > maximum:
        raise SkillRefused(f"Ça ferait {len(etapes)} étapes, au-delà de la limite de {maximum}.")

    fonction = _nom_de_fichier(titre)
    chemin, existait = _ecrire_le_plugin_de_routine(titre, etapes, fonction, ORIGINE_COMPOSEE)
    get_logger().info("Capacité composée : %s (%d étapes)", chemin.name, len(etapes))

    resume = _resumer_les_etapes(etapes)
    return {
        "speak": f"C'est fait. Dites « {titre} » et j'enchaînerai : {resume}."
                 + (" J'ai remplacé la routine du même nom." if existait else ""),
        "display": f"{chemin}\n\n" + "\n".join(
            f"  {index}. {_decrire_etape(competence, arguments)}"
            for index, (competence, arguments) in enumerate(etapes, 1)
        ) + "\n\nLe rechargement à chaud la prendra d'ici une seconde.",
    }


# --- en demander une neuve : proposer, relire, puis seulement installer -----

@skill(
    description="Écrit une proposition de nouvelle capacité en Python, à réviser avant de l'installer.",
    examples=[
        "écris une capacité qui convertit des devises",
        "crée-moi une nouvelle capacité pour deviner la météo",
        "invente une compétence qui traduit une phrase",
    ],
    timeout=120.0,
)
def proposer_une_capacite(nom: str, description: str) -> dict:
    """Fait écrire un plugin complet par le modèle local, sans l'activer.

    Le fichier part dans un dossier à l'écart de ``plugins/`` : le
    surveillant ne le voit jamais, rien ne se charge tout seul. Dites
    « active la capacité <nom> » après l'avoir relu pour l'installer, ou
    « oublie la proposition <nom> » pour la jeter.

    Args:
        nom: Le nom donné à la capacité proposée.
        description: Ce qu'elle doit faire.
    """
    titre = _nettoyer_le_nom(nom) or nom.strip()
    if not titre:
        raise SkillRefused("Donnez-lui un nom : « écris une capacité, convertisseur de devises ».")
    description = description.strip()
    if not description:
        raise SkillRefused("Que doit faire cette capacité ?")

    code = _nettoyer_le_code(demander_au_modele(
        description, system=CONSIGNE_CAPACITE,
        max_tokens=int(get_config("max_tokens_capacite", 900)), temperature=0.1,
    ))
    if "@skill" not in code:
        raise SkillRefused("Le modèle n'a pas produit un plugin exploitable ; reformulez la demande.")
    try:
        compile(code, "<proposition>", "exec")
    except SyntaxError as exc:
        raise SkillRefused(f"Le code proposé ne compile pas : {exc}") from None

    fonction = _nom_de_fichier(titre)
    chemin = _dossier_des_propositions() / f"{fonction}.py"
    chemin.write_text(code, encoding="utf-8")
    get_logger().info("Capacité proposée, en attente de relecture : %s", chemin)

    avertissements = []
    risques = _jetons_risques_presents(code)
    if risques:
        avertissements.append(f"j'y vois {', '.join(risques)} : relisez particulièrement ce passage")
    if (dossier_des_plugins() / f"{fonction}.py").exists():
        avertissements.append("l'installer remplacerait un fichier existant de plugins/ (une sauvegarde serait gardée)")

    parle = f"J'ai écrit une proposition pour « {titre} ». Relisez-la, puis dites « active la capacité {titre} »."
    if avertissements:
        parle += " Attention : " + " ; ".join(avertissements) + "."
    return {"speak": parle, "display": f"{chemin}\n\n{code}"}


@skill(
    description="Liste les capacités proposées par le modèle, en attente de relecture.",
    examples=["quelles capacités sont en attente", "mes propositions de capacités"],
)
def mes_propositions() -> dict:
    """Les fichiers écrits par ``proposer_une_capacite``, pas encore installés."""
    fichiers = sorted(_dossier_des_propositions().glob("*.py"))
    if not fichiers:
        return {"speak": "Aucune capacité en attente de relecture.", "display": "aucune proposition"}
    noms = [fichier.stem for fichier in fichiers]
    return {
        "speak": f"{len(fichiers)} proposition{'s' if len(fichiers) > 1 else ''} en attente : "
                 + ", ".join(noms) + ".",
        "display": "\n".join(str(fichier) for fichier in fichiers),
    }


@skill(
    description="Installe dans plugins/ une capacité proposée par le modèle, après relecture.",
    examples=["active la capacité convertisseur de devises", "installe cette proposition"],
    confirm="Vous l'avez relue ? Voulez-vous vraiment l'installer ?",
)
def activer_la_capacite_proposee(nom: str) -> str:
    """Déplace une proposition relue dans plugins/, en sauvegardant l'existant.

    Revalide le code avant de le déplacer : rien ne garantit qu'il n'a pas été
    édité entre la proposition et cette confirmation, en bien ou en mal.

    Args:
        nom: Le nom donné à la proposition.
    """
    fonction = _nom_de_fichier(nom)
    source = _dossier_des_propositions() / f"{fonction}.py"
    if not source.exists():
        raise SkillRefused(f"Je n'ai pas de proposition « {nom} » en attente.")

    code = source.read_text(encoding="utf-8")
    try:
        compile(code, str(source), "exec")
    except SyntaxError as exc:
        raise SkillRefused(f"Cette proposition ne compile plus : {exc}") from None
    risques = _jetons_risques_presents(code)
    if risques:
        raise SkillRefused(
            f"Cette proposition contient {', '.join(risques)} : je ne l'installe pas "
            "automatiquement. Éditez le fichier pour l'en retirer, ou installez-le vous-même."
        )

    destination = dossier_des_plugins() / f"{fonction}.py"
    if destination.exists():
        horodatage = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(destination, destination.with_name(f"{destination.name}.{horodatage}.sauvegarde"))
    shutil.move(str(source), str(destination))
    get_logger().info("Capacité installée depuis une proposition : %s", destination)
    return f"Capacité « {nom} » installée. Le rechargement à chaud la prendra d'ici une seconde."


@skill(
    description="Supprime une capacité proposée par le modèle qui n'a pas été installée.",
    examples=["oublie cette proposition", "jette la proposition convertisseur"],
)
def rejeter_la_proposition(nom: str) -> str:
    """Efface le fichier d'une proposition en attente.

    Args:
        nom: Le nom donné à la proposition à écarter.
    """
    fonction = _nom_de_fichier(nom)
    source = _dossier_des_propositions() / f"{fonction}.py"
    if not source.exists():
        raise SkillRefused(f"Je n'ai pas de proposition « {nom} » en attente.")
    source.unlink()
    return f"La proposition « {nom} » est écartée."


# --- lister et oublier, quelle que soit l'origine ----------------------------

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
                     "puis dites « retiens cette routine », ou décrivez-en une.",
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

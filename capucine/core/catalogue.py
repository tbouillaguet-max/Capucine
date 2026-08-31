"""Le catalogue d'API : ce que vos fonctions savent faire, dit au modèle.

Le mode d'échec numéro un d'un petit modèle sur VOTRE dépôt n'est pas la
syntaxe, c'est l'**invention d'API**. Il ne sait pas que
``capped_weights(conviction, cap_pct=None)`` existe, alors il écrit
``normalize_weights()`` — plausible, inexistante, et le code ne tourne pas.

Ce module lit les signatures et les docstrings de vos fichiers, et en fait une
référence compacte à glisser dans le contexte. Donnez la signature au modèle,
il ne peut plus l'inventer de travers.

C'est le mécanisme de ``schema.py`` — signature Python plus docstring égale
contrat pour le modèle — appliqué cette fois au code que vous écrivez, et
plus seulement aux compétences.

Trois choix de conception
-------------------------
* **Lecture par AST, jamais par import.** Importer le dépôt exécuterait son
  code au niveau module et exigerait ses dépendances installées. On lit la
  source comme du texte : aucun effet de bord, et un fichier qui ne
  s'importerait pas se catalogue quand même.
* **Sélection, pas déversement.** On ne met pas quatre cents signatures dans
  un 7B. On en choisit une quinzaine, et le reste attend son tour.
* **Le résumé compte autant que le nom.** Les identifiants sont en anglais,
  vos docstrings en français : chercher sur les deux à la fois est ce qui
  permet à « poids plafonnés » de retrouver ``capped_weights``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .logging import get_logger
from .schema import parse_docstring
from .text import normalize, similarity

logger = get_logger("catalogue")

IGNORES = {
    "__pycache__", ".git", ".venv", "venv", "node_modules", "build", "dist",
    ".pytest_cache", ".ruff_cache", ".mypy_cache",
}
# Les tests sont écartés par défaut, et ce n'est pas de la commodité : le
# catalogue répond à « que puis-je appeler ». Personne n'appelle
# `test_le_plafond_par_position_reste_applique` depuis du code neuf — mais
# son nom, lui, capte toutes les recherches sur « plafond par position ».
DOSSIERS_DE_TESTS = {"tests", "test", "testing"}


def _est_un_test(chemin: Path) -> bool:
    if any(partie in DOSSIERS_DE_TESTS for partie in chemin.parts):
        return True
    nom = chemin.name
    return nom.startswith("test_") or nom.endswith("_test.py") or nom == "conftest.py"


# Au-delà de quel écart au bruit de fond une entrée mérite d'être montrée.
DETACHEMENT = 1.6
POPULATION_MINIMALE = 12
# Le plancher se lit dans les mesures, il ne se devine pas. Sur un dépôt réel
# de 400 entrées, avec les VRAIES descriptions du banc : cinq demandes sans
# rapport plafonnent à 0,233, les demandes qui ont leur réponse marquent 0,44
# et 0,61. On se pose entre les deux. Refaites la mesure si vous changez le
# calcul du score — c'est une valeur empirique, pas une constante.
#
# Une limite connue et assumée : « en plafonnant chaque poids » ne retrouve
# pas `capped_weights`, dont le résumé dit « aucun ne dépassant cap_pct % ».
# Les deux disent la même chose sans partager un mot : c'est un rapprochement
# SÉMANTIQUE, hors de portée d'un score lexical, et c'est là que l'index de
# semantique.py gagnerait sa place.
PLANCHER = 0.30

# Le contenu entre parenthèses d'un résumé est presque toujours un
# qualificatif, jamais la définition : « (config.BACKTEST_MAX_WEIGHT_PER_
# POSITION_PCT par défaut) » ajoute huit jetons que personne ne prononcera,
# et dilue d'autant la couverture de l'entrée.
_SANS_PARENTHESE = re.compile(r"\([^)]*\)")
_IDENTIFIANT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _jetons_utiles(texte: str) -> set[str]:
    return {mot for mot in texte.split() if len(mot) > 2 and mot not in MOTS_VIDES}

# Les mots outils n'apportent aucun signal et apparient partout : sans ce
# filtre, « les » et « par » offrent le bonus de jetons à la moitié du dépôt.
MOTS_VIDES = {
    "les", "des", "une", "aux", "mon", "ton", "son", "mes", "tes", "ses",
    "par", "pour", "avec", "dans", "sur", "sous", "que", "qui", "quoi",
    "est", "sont", "cette", "cet", "ces", "leur", "nos", "vos", "puis",
    "the", "and", "for", "with", "from", "into", "this", "that",
}


def _au_dessus_du_fond(classees: list) -> list:
    """Ne garde que ce qui se détache du bruit — quitte à ne rien garder.

    Un seuil absolu ne peut pas trancher : les scores vivent tous dans une
    bande étroite, et « additionne deux entiers » y obtient 0,26 quand
    « poids plafonnés » obtient 0,36. Ce qui sépare les deux n'est pas le
    niveau, c'est la FORME. Une demande qui a une réponse dans le dépôt
    produit un pic ; une demande hors sujet produit un plateau.

    On mesure donc le bruit de fond — la médiane des mieux classés — et on ne
    retient que ce qui s'en détache. Sur une demande sans rapport, rien ne
    passe, et c'est le bon résultat : injecter douze fonctions au hasard dans
    le contexte d'un 7B est pire que de n'en injecter aucune.
    """
    if not classees:
        return []
    seuil = PLANCHER
    # Le bruit de fond suppose une population : sur une poignée d'entrées, la
    # médiane EST le sommet, et le test relatif rejetterait jusqu'à la bonne
    # réponse. En dessous de ce seuil de population, seul le plancher joue.
    if len(classees) >= POPULATION_MINIMALE:
        scores = [score for score, _ in classees]
        echantillon = scores[: min(50, len(scores))]
        fond = echantillon[len(echantillon) // 2]  # la médiane, liste déjà triée
        seuil = max(PLANCHER, fond * DETACHEMENT)
    return [(score, entree) for score, entree in classees if score >= seuil]


def _apparie(mot: str, jetons: set[str]) -> float:
    """À quel point ce mot se retrouve dans le lot — de 0 à 1.

    Trois degrés, du plus sûr au plus permissif :

    1. le mot exact ;
    2. le préfixe — « option » trouve « options », « valoris » trouve
       « valorisation » ;
    3. le **cognat** — « volatilite » trouve « volatility ». C'est le cas qui
       compte ici : les identifiants de ce dépôt sont en anglais, ses
       docstrings et vos demandes en français, et les deux langues partagent
       la moitié de leur vocabulaire technique.

    Le cognat n'est tenté que si les trois premières lettres coïncident : ça
    évite de comparer chaque mot à tout le dépôt, et un cognat qui ne
    partagerait pas son début n'en est pas un.
    """
    if mot in jetons:
        return 1.0
    if len(mot) < 4:
        return 0.0
    meilleur = 0.0
    for jeton in jetons:
        if len(jeton) < 4:
            continue
        if jeton.startswith(mot) or mot.startswith(jeton):
            return 0.85
        if jeton[:3] != mot[:3]:
            continue
        # Un long préfixe commun est un signal plus sûr qu'un ratio global
        # abaissé : « plafonnant » et « plafonne » partagent sept lettres,
        # mais leurs terminaisons divergentes font tomber le ratio sous 0,80.
        if _prefixe_commun(mot, jeton) >= 6:
            return 0.8
        ratio = SequenceMatcher(None, mot, jeton).ratio()
        if ratio > meilleur:
            meilleur = ratio
    return meilleur if meilleur >= 0.8 else 0.0


def _prefixe_commun(gauche: str, droite: str) -> int:
    longueur = 0
    for a, b in zip(gauche, droite, strict=False):
        if a != b:
            break
        longueur += 1
    return longueur


def _mots(texte: str) -> str:
    """Normalise en traitant le tiret bas comme une coupure de mot.

    `normalize` le garde — c'est un caractère de mot pour `\\w` — si bien que
    « capped weights » ne se retrouvait pas dans « utilise capped_weights ».
    """
    return normalize((texte or "").replace("_", " "))


@dataclass
class Entree:
    """Une fonction, une méthode ou une classe, telle qu'on la montre au modèle."""

    nom: str                  # capped_weights
    qualifie: str             # backtest.strategies.base.capped_weights
    signature: str            # (conviction: pd.Series, cap_pct: float | None = None) -> pd.Series
    resume: str               # la première phrase de la docstring
    genre: str                # fonction | methode | classe
    fichier: str              # chemin relatif dans le dépôt
    ligne: int

    @property
    def module(self) -> str:
        """Le module, sans le nom de la fonction ni celui de la classe."""
        morceaux = self.qualifie.split(".")
        retire = 2 if self.genre == "methode" else 1
        return ".".join(morceaux[:-retire])

    @property
    def racine(self) -> str:
        """Ce qu'il faut importer : la classe pour une méthode, sinon le nom."""
        return self.qualifie.split(".")[-2] if self.genre == "methode" else self.nom

    def importation(self) -> str:
        """La ligne d'import à écrire — ou pourquoi il n'y en a pas.

        Tous les modules ne s'importent pas : « 08_recuperation_options »
        commence par un chiffre, `from 08_… import x` est une erreur de
        syntaxe. Montrer un import impossible apprendrait au modèle à en
        écrire un, ce qui est pire que de n'en montrer aucun.
        """
        if self.module and all(_IDENTIFIANT.fullmatch(part) for part in self.module.split(".")):
            return f"from {self.module} import {self.racine}"
        return f"# défini dans {self.fichier}"

    def appel(self) -> str:
        """La forme de l'appel, en Python VALIDE.

        `def backtest.strategies.base.capped_weights(...)` ne se déclare pas
        et ne s'appelle pas : c'est du Python impossible. Un modèle de code
        entraîné sur du Python suit une forme qu'il reconnaît, pas une
        notation pointée inventée pour l'occasion.
        """
        if self.genre == "classe":
            return f"{self.nom}{self.signature or '()'}"
        if self.genre == "methode":
            return f"{self.racine}.{self.nom}{self.signature}"
        return f"{self.nom}{self.signature}"

    def declaration(self) -> str:
        return f"{self.importation()}\n{self.appel()}"

    def rendu(self) -> str:
        """Ce que le modèle voit : comment l'importer, comment l'appeler,
        et à quoi elle sert."""
        if not self.resume:
            return self.declaration()
        return f"{self.declaration()}\n    → {self.resume}"

    @property
    def _cherchable(self) -> str:
        # Nom ET résumé : les identifiants sont en anglais, les docstrings de
        # ce projet en français. Chercher sur un seul des deux rate la moitié.
        return f"{self.nom.replace('_', ' ')} {self.qualifie} {self.resume}"


class Catalogue:
    """Les signatures d'un dépôt, lues par AST et tenues à jour."""

    def __init__(
        self,
        racine: str | Path,
        *,
        entrees_rendues: int = 15,
        caracteres_max: int = 3000,
        privees: bool = False,
        tests: bool = False,
    ) -> None:
        self.racine = Path(racine).expanduser()
        self.entrees_rendues = entrees_rendues
        self.caracteres_max = caracteres_max
        self.privees = privees
        self.tests = tests
        self._entrees: list[Entree] = []
        # (chemin, date) des fichiers lus : une relecture ne coûte que si un
        # fichier a bougé.
        self._empreintes: dict[Path, float] = {}

    # -- construction -------------------------------------------------------
    def construire(self, force: bool = False) -> int:
        """(Re)lit les fichiers modifiés. Rend le nombre d'entrées connues."""
        fichiers = self._fichiers()
        empreintes = {chemin: chemin.stat().st_mtime for chemin in fichiers}
        if not force and empreintes == self._empreintes and self._entrees:
            return len(self._entrees)

        entrees: list[Entree] = []
        illisibles = 0
        for chemin in fichiers:
            try:
                entrees.extend(self._lire(chemin))
            except (OSError, SyntaxError, ValueError):
                # Un fichier abîmé ou d'une autre version de Python ne doit pas
                # priver le modèle de tout le reste du dépôt.
                illisibles += 1
        self._entrees = entrees
        self._empreintes = empreintes
        logger.info(
            "Catalogue de %s : %d entrées dans %d fichiers%s",
            self.racine.name, len(entrees), len(fichiers),
            f", {illisibles} illisible(s)" if illisibles else "",
        )
        return len(entrees)

    def _fichiers(self) -> list[Path]:
        return sorted(
            chemin for chemin in self.racine.rglob("*.py")
            if not any(partie in IGNORES for partie in chemin.parts)
            and (self.tests or not _est_un_test(chemin.relative_to(self.racine)))
        )

    def _lire(self, chemin: Path) -> list[Entree]:
        source = chemin.read_text(encoding="utf-8", errors="replace")
        arbre = ast.parse(source, filename=str(chemin))
        relatif = chemin.relative_to(self.racine).as_posix()
        prefixe = relatif[:-3].replace("/", ".")
        entrees: list[Entree] = []
        for noeud in arbre.body:
            entrees.extend(self._entrees_du_noeud(noeud, prefixe, relatif))
        return entrees

    def _entrees_du_noeud(self, noeud: ast.AST, prefixe: str, relatif: str) -> list[Entree]:
        if isinstance(noeud, ast.ClassDef):
            entrees = [self._entree(noeud, prefixe, relatif, "classe")] if self._garder(noeud.name) else []
            for enfant in noeud.body:
                if isinstance(enfant, ast.FunctionDef | ast.AsyncFunctionDef) and self._garder(enfant.name):
                    entrees.append(
                        self._entree(enfant, f"{prefixe}.{noeud.name}", relatif, "methode")
                    )
            return entrees
        if isinstance(noeud, ast.FunctionDef | ast.AsyncFunctionDef) and self._garder(noeud.name):
            return [self._entree(noeud, prefixe, relatif, "fonction")]
        return []

    def _garder(self, nom: str) -> bool:
        if self.privees:
            return True
        # `__init__` est une exception : c'est la signature de construction,
        # et sans elle on ne sait pas comment instancier la classe.
        return not nom.startswith("_") or nom == "__init__"

    def _entree(self, noeud, prefixe: str, relatif: str, genre: str) -> Entree:
        resume, _ = parse_docstring(ast.get_docstring(noeud))
        return Entree(
            nom=noeud.name,
            qualifie=f"{prefixe}.{noeud.name}" if prefixe else noeud.name,
            signature=_signature(noeud) if genre != "classe" else _signature_de_classe(noeud),
            resume=_premiere_phrase(resume),
            genre=genre,
            fichier=relatif,
            ligne=getattr(noeud, "lineno", 0),
        )

    # -- consultation -------------------------------------------------------
    def chercher(self, question: str, limite: int = 0) -> list[Entree]:
        """Les entrées les plus proches de la demande, les meilleures d'abord."""
        self.construire()
        limite = limite or self.entrees_rendues
        question = (question or "").strip()
        if not question or not self._entrees:
            return []

        demande = _mots(question)
        mots = _jetons_utiles(demande)
        classees = [(self._noter(demande, mots, entree), entree) for entree in self._entrees]

        classees.sort(key=lambda paire: paire[0], reverse=True)
        return [entree for _, entree in _au_dessus_du_fond(classees)[:limite]]

    def _noter(self, demande: str, mots: set[str], entree: Entree) -> float:
        """Deux mesures mêlées, parce qu'aucune ne suffit seule.

        La **couverture** demande : quelle part de l'ENTRÉE la demande
        recouvre-t-elle ? Elle est asymétrique, donc insensible à la longueur
        de la demande — et c'est tout le point. La mesure symétrique d'avant
        pénalisait une description longue et naturelle (« écris une fonction
        qui pondère un dictionnaire… en plafonnant chaque poids »),
        c'est-à-dire exactement ce qu'on écrit vraiment : sur les vraies
        tâches du banc, elle ne trouvait rien pour deux demandes sur trois.

        La **similarité** symétrique rattrape les demandes courtes, où la
        couverture est fragile. Le mélange bat chacune prise seule, mesuré
        sur les mêmes demandes.
        """
        cherchable = _mots(entree._cherchable)
        jetons_texte = set(cherchable.split())
        symetrique = similarity(demande, cherchable)
        communs = sum(_apparie(mot, jetons_texte) for mot in mots)
        if communs:
            symetrique += 0.35 * communs / max(1, len(mots))

        resume = _SANS_PARENTHESE.sub(" ", entree.resume)
        cibles = _jetons_utiles(_mots(f"{entree.nom} {resume}"))
        couverture = (
            sum(_apparie(cible, mots) for cible in cibles) / len(cibles) if cibles else 0.0
        )
        score = 0.65 * couverture + 0.35 * symetrique
        # Nommer une fonction explicitement doit la sortir en tête, quel que
        # soit le reste de la phrase.
        if _mots(entree.nom) in demande:
            score += 1.0
        return score

    def reference(self, question: str, limite: int = 0) -> str:
        """La référence d'API à glisser dans le contexte, bornée en taille."""
        entrees = self.chercher(question, limite)
        if not entrees:
            return ""
        lignes: list[str] = []
        total = 0
        for entree in entrees:
            rendu = entree.rendu()
            # La meilleure entrée passe toujours, même si elle dépasse à elle
            # seule le budget : rendre une référence VIDE parce que la
            # réponse est un peu longue serait le pire des deux mondes. On la
            # réduit alors à sa déclaration, qui est l'essentiel.
            if total + len(rendu) > self.caracteres_max:
                if lignes:
                    break
                rendu = entree.declaration()[: self.caracteres_max]
            lignes.append(rendu)
            total += len(rendu)
        return "\n\n".join(lignes)

    def par_nom(self, nom: str) -> Entree | None:
        self.construire()
        vise = nom.strip().lower()
        for entree in self._entrees:
            qualifie = entree.qualifie.lower()
            # Un suffixe suffit : on veut pouvoir demander
            # « Portefeuille.valoriser » sans connaître le chemin du module.
            if entree.nom.lower() == vise or qualifie == vise or qualifie.endswith(f".{vise}"):
                return entree
        return None

    def statistiques(self) -> dict[str, int]:
        self.construire()
        compte: dict[str, int] = {"total": len(self._entrees), "fichiers": len(self._empreintes)}
        for entree in self._entrees:
            compte[entree.genre] = compte.get(entree.genre, 0) + 1
        return compte


# --- rendu des signatures ---------------------------------------------------

def _signature(noeud) -> str:
    """La signature telle qu'elle est écrite, annotations comprises.

    On repasse par ``ast.unparse`` plutôt que par le texte brut : une
    signature écrite sur six lignes redevient une ligne lisible.
    """
    args = noeud.args
    morceaux: list[str] = []

    obligatoires = args.posonlyargs + args.args
    defauts = [None] * (len(obligatoires) - len(args.defaults)) + list(args.defaults)
    for index, argument in enumerate(obligatoires):
        if argument.arg in ("self", "cls") and index == 0:
            continue
        morceaux.append(_argument(argument, defauts[index]))
        if args.posonlyargs and index == len(args.posonlyargs) - 1:
            morceaux.append("/")

    if args.vararg:
        morceaux.append(f"*{_argument(args.vararg, None)}")
    elif args.kwonlyargs:
        morceaux.append("*")
    for argument, defaut in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        morceaux.append(_argument(argument, defaut))
    if args.kwarg:
        morceaux.append(f"**{_argument(args.kwarg, None)}")

    retour = f" -> {_texte(noeud.returns)}" if noeud.returns else ""
    return f"({', '.join(morceaux)}){retour}"


def _signature_de_classe(noeud: ast.ClassDef) -> str:
    bases = [_texte(base) for base in noeud.bases]
    return f"({', '.join(bases)})" if bases else ""


def _argument(argument: ast.arg, defaut) -> str:
    rendu = argument.arg
    if argument.annotation is not None:
        rendu += f": {_texte(argument.annotation)}"
    if defaut is not None:
        rendu += f" = {_texte(defaut)}"
    return rendu


def _texte(noeud) -> str:
    try:
        return ast.unparse(noeud)
    except Exception:  # pragma: no cover - une annotation exotique
        return "…"


def _premiere_phrase(resume: str, maximum: int = 160) -> str:
    """Une phrase suffit : le modèle a besoin du contrat, pas du traité."""
    resume = " ".join((resume or "").split())
    if not resume:
        return ""
    if ". " in resume[:maximum]:
        return resume[: resume.index(". ") + 1]
    for separateur in (" : ", " — "):
        coupe = resume.find(separateur)
        # Une coupe très tôt n'est pas une fin de phrase mais un nom de
        # paramètre en tête de docstring (« signals : signaux connus… ») :
        # la garder rendrait un résumé qui ne dit rien.
        if 30 <= coupe < maximum:
            return resume[:coupe].rstrip()
    return resume[:maximum].rstrip()


def depuis_config(config, racine: str | Path | None = None) -> Catalogue | None:
    """Construit le catalogue décrit par ``[catalogue]``."""
    section = config.section("catalogue")
    if not bool(section.get("active", True)):
        logger.info("Catalogue d'API désactivé par la configuration.")
        return None
    chemin = racine or config.resolve_path("catalogue.racine")
    if chemin is None:
        return None
    return Catalogue(
        chemin,
        entrees_rendues=int(section.get("entrees_rendues", 15)),
        caracteres_max=int(section.get("caracteres_max", 3000)),
        privees=bool(section.get("privees", False)),
        tests=bool(section.get("tests", False)),
    )

"""Piloter CalculRisque_Mark5 à la voix : mises à jour, optimisations, stratégies.

``projet.py`` sait lancer *n'importe quel* projet et rendre compte. Ce
plugin-ci en sait davantage sur **celui-là** : ses enchaînements nommés, ses
quatre optimiseurs, et la forme exacte d'une stratégie dans son registre.
C'est la division voulue par le contrat — le générique reste générique, le
spécialisé tient dans un fichier qu'on peut lire et modifier.

Il ne partage pas le lanceur de tâches de fond de ``projet.py``, et ce n'est
pas un oubli : un plugin qui en importe un autre casse au premier
rechargement à chaud de l'importé. Les deux lanceurs diffèrent d'ailleurs —
celui-là enchaîne **plusieurs étapes** et s'arrête à la première qui échoue.

Ce qu'elle ne fait pas
----------------------
Elle **n'écrit jamais** dans un fichier existant du dépôt, sauf la seule ligne
d'import qui enregistre une nouvelle stratégie — et avec sauvegarde. Ni les
scripts de valorisation, ni le moteur de backtest, ni ``config.py`` ne sont
touchés. Une stratégie créée sort d'un **gabarit vérifié**, pas du modèle de
langage : un 7B quantifié qui écrit de la logique de valorisation produit du
code plausible et silencieusement faux, et il s'agit ici d'argent.

Déclaration, dans ``config/pc.toml`` ::

    [plugins.calculrisque]
    chemin = "~/projets/CalculRisque_Mark5"
    delai_s = 21600
    variables = { SEC_CONTACT_EMAIL = "vous@exemple.fr" }

    [plugins.calculrisque.enchainements]
    quotidienne = ["03b_recuperation_cours_quotidiens.py"]
    trimestrielle = ["run_pipeline_quarterly.py --resume"]

    [plugins.calculrisque.optimisations]
    stops = "11_optimize_options_stops.py"
"""

import csv
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from capucine.plugin import (
    SkillRefused,
    announce,
    atelier,
    get_config,
    get_logger,
    skill,
)

CONFIG_DEFAULTS = {
    "chemin": "",
    "delai_s": 21600.0,          # six heures : une optimisation en prend plusieurs
    "lignes_journal": 20,
    "variables": {},
    "enchainements": {
        "quotidienne": ["03b_recuperation_cours_quotidiens.py"],
        "trimestrielle": ["run_pipeline_quarterly.py --resume"],
    },
    "optimisations": {
        "stops": "11_optimize_options_stops.py",
        "rebalancement": "11b_optimize_rebalance_threshold.py",
        "convergence": "11c_optimize_convergence_fraction.py",
        "seuil_entree": "11d_optimize_entry_threshold.py",
        "multiples": "optimize_options_multiples.py",
    },
    "dossier_optimisations": "data/optimisations",
    "strategie_par_defaut": "valuation_gap_options",
}

# Un seul travail de fond à la fois : deux optimisations en parallèle se
# disputeraient les cœurs, et deux mises à jour la même base de données.
_TRAVAIL: dict = {}
_VERROU = threading.Lock()


def on_unload() -> None:
    """Coupe le travail en cours avant que le module ne disparaisse."""
    with _VERROU:
        processus = _TRAVAIL.get("processus")
        if processus is not None and processus.poll() is None:
            get_logger().info("Arrêt du travail CalculRisque au déchargement.")
            processus.kill()
        _TRAVAIL.clear()


# --- le dépôt ---------------------------------------------------------------

def _depot() -> Path:
    """Le dossier du projet, obligatoirement dans l'atelier.

    L'atelier est la frontière : tant que vous ne l'avez pas ouvert, ce plugin
    est inerte. C'est voulu — il lance des programmes et écrit des fichiers.
    """
    chemin = str(get_config("chemin", "") or "").strip()
    if not chemin:
        raise SkillRefused(
            "Aucun dépôt CalculRisque déclaré. Renseignez "
            "plugins.calculrisque.chemin dans la configuration."
        )
    dossier = atelier().resoudre(chemin, doit_exister=True)
    if not dossier.is_dir():
        raise SkillRefused(f"« {dossier} » n'est pas un dossier.")
    return dossier


def _environnement() -> dict:
    env = dict(os.environ)
    env.update({str(c): str(v) for c, v in (get_config("variables", {}) or {}).items()})
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _occupe() -> str:
    """Un travail est-il en cours ? Le verdict fait foi, pas le processus.

    Interroger `processus.poll()` se trompait des deux côtés : juste après
    « lance », le sous-processus n'est pas encore né et la place semblait
    libre — deux commandes rapprochées démarraient toutes les deux ; juste
    après « arrête », `kill()` étant asynchrone, elle semblait encore prise.
    Un travail est en cours du moment où il est inscrit jusqu'à ce qu'un
    verdict soit rendu, et c'est tout.
    """
    with _VERROU:
        if not _TRAVAIL or _TRAVAIL.get("fini"):
            return ""
        return str(_TRAVAIL.get("libelle", "un travail"))


# --- l'exécution en fond ----------------------------------------------------

def _lancer(libelle: str, etapes: list[str], sortie: Path | None = None) -> None:
    """Enchaîne des commandes en tâche de fond, et annonce le verdict.

    On rend la main tout de suite : une optimisation tourne des heures, et un
    tour de parole ne se garde pas des heures.
    """
    occupe = _occupe()
    if occupe:
        raise SkillRefused(
            f"{occupe} est déjà en cours. Demandez « où en est le pipeline », "
            "ou arrêtez-le d'abord."
        )
    depot = _depot()
    with _VERROU:
        jeton = int(_TRAVAIL.get("jeton", 0)) + 1
        _TRAVAIL.clear()
        _TRAVAIL.update({
            "jeton": jeton, "libelle": libelle, "etapes": list(etapes), "etape": 0,
            "debut": time.monotonic(), "journal": [], "processus": None,
            "sortie": str(sortie) if sortie else "", "fini": None, "arret": False,
        })
    fil = threading.Thread(
        target=_derouler, args=(jeton, libelle, etapes, depot), daemon=True,
        name=f"calculrisque-{libelle}",
    )
    fil.start()


def _toujours_le_notre(jeton: int) -> bool:
    """Ce travail est-il encore celui en cours, et non arrêté ?

    Le jeton évite qu'un fil resté en arrière n'écrive dans le travail
    suivant : sans lui, une étape lente d'un travail arrêté viendrait
    renuméroter les étapes de celui qu'on vient de lancer.
    """
    return bool(_TRAVAIL) and _TRAVAIL.get("jeton") == jeton and not _TRAVAIL.get("arret")


def _derouler(jeton: int, libelle: str, etapes: list[str], depot: Path) -> None:
    """Enchaîne les étapes. Rend toujours un verdict, quoi qu'il arrive.

    L'invariant tient toute la logique d'occupation : un travail sans verdict
    garderait la place indéfiniment.
    """
    try:
        _derouler_vraiment(jeton, libelle, etapes, depot)
    except Exception:  # pragma: no cover - un imprévu ne bloque pas la place
        get_logger().exception("Le déroulement de %s a échoué.", libelle)
        _terminer(jeton, libelle, f"{libelle} : interrompu par une erreur interne.")


def _derouler_vraiment(jeton: int, libelle: str, etapes: list[str], depot: Path) -> None:
    delai = float(get_config("delai_s", 21600.0))
    environnement = _environnement()
    for numero, commande in enumerate(etapes, 1):
        with _VERROU:
            if not _toujours_le_notre(jeton):
                return          # arrêté, déchargé, ou remplacé entre deux étapes
            _TRAVAIL["etape"] = numero
        argv = [sys.executable, *shlex.split(commande)]
        get_logger().info("[%s] étape %d/%d : %s", libelle, numero, len(etapes), commande)
        try:
            processus = subprocess.Popen(
                argv, cwd=str(depot), env=environnement,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
        except OSError as erreur:
            _terminer(jeton, libelle, f"L'étape {numero} n'a pas pu démarrer : {erreur}")
            return
        with _VERROU:
            if not _toujours_le_notre(jeton):
                processus.kill()      # arrêté pendant qu'il naissait
                return
            _TRAVAIL["processus"] = processus
        code = _suivre_la_sortie(jeton, processus, delai)
        if code != 0:
            _terminer(
                jeton, libelle,
                f"L'étape {numero} sur {len(etapes)} a échoué (code {code}). "
                "Demandez-moi son journal.",
            )
            return
    _terminer(jeton, libelle, f"{libelle} : terminé, {len(etapes)} étape(s) réussie(s).")


def _suivre_la_sortie(jeton: int, processus, delai: float) -> int:
    """Lit la sortie ligne par ligne et borne la durée. Rend le code de retour."""
    limite = time.monotonic() + delai
    maximum = int(get_config("lignes_journal", 20)) * 6
    if processus.stdout is not None:
        for ligne in processus.stdout:
            with _VERROU:
                if not _toujours_le_notre(jeton):
                    processus.kill()
                    return -1
                journal = _TRAVAIL["journal"]
                journal.append(ligne.rstrip())
                del journal[:-maximum]
            if time.monotonic() > limite:
                processus.kill()
                get_logger().error("Étape abandonnée : délai de %.0f s dépassé.", delai)
                return -9
    return processus.wait()


def _terminer(jeton: int, libelle: str, phrase: str) -> None:
    with _VERROU:
        if _TRAVAIL.get("jeton") != jeton:
            return                      # un autre travail a pris la place
        _TRAVAIL["fini"] = phrase
        _TRAVAIL["processus"] = None
    get_logger().info("[%s] %s", libelle, phrase)
    announce(phrase)


# --- compétences : les enchaînements ----------------------------------------

@skill(
    description="Lance la mise à jour des données de marché du projet CalculRisque.",
    examples=[
        "fais la mise à jour quotidienne",
        "mets à jour les cours",
        "lance la mise à jour trimestrielle",
        "actualise les données du projet",
    ],
)
def mettre_a_jour(quoi: str = "quotidienne") -> str:
    """Enchaîne les étapes déclarées pour cette mise à jour, en tâche de fond.

    Args:
        quoi: L'enchaînement à lancer, tel que déclaré dans la configuration
            (« quotidienne », « trimestrielle »…).
    """
    enchainements = dict(get_config("enchainements", {}) or {})
    clef = _rapprocher(quoi, enchainements)
    if clef is None:
        raise SkillRefused(
            f"Je ne connais pas la mise à jour « {quoi} ». "
            f"J'ai : {', '.join(sorted(enchainements)) or 'aucune'}."
        )
    etapes = [str(etape) for etape in enchainements[clef]]
    if not etapes:
        raise SkillRefused(f"L'enchaînement « {clef} » ne déclare aucune étape.")
    _lancer(f"mise à jour {clef}", etapes)
    return (
        f"C'est parti : mise à jour {clef}, {len(etapes)} étape"
        f"{'s' if len(etapes) > 1 else ''}. Je vous préviens quand c'est fini."
    )


@skill(
    description="Dit où en est le travail CalculRisque en cours et depuis combien de temps.",
    examples=[
        "où en est le pipeline",
        "où en est la mise à jour",
        "où en est l'optimisation",
        "état du projet calculrisque",
    ],
)
def etat_du_travail() -> dict:
    """L'étape courante, la durée écoulée, et la fin du journal."""
    with _VERROU:
        travail = dict(_TRAVAIL)
        journal = list(travail.get("journal", []))
    if not travail:
        return {"speak": "Aucun travail en cours sur CalculRisque.", "display": "au repos"}

    minutes = (time.monotonic() - travail["debut"]) / 60.0
    lignes = journal[-int(get_config("lignes_journal", 20)):]
    fini = travail.get("fini")
    if fini:
        return {"speak": fini, "display": f"{fini}\n\n" + "\n".join(lignes)}
    return {
        "speak": f"{travail['libelle']} : étape {travail['etape']} sur "
                 f"{len(travail['etapes'])}, depuis {minutes:.0f} minutes.",
        "display": (
            f"{travail['libelle']} — étape {travail['etape']}/{len(travail['etapes'])}"
            f" — {minutes:.1f} min\n"
            f"commande : {travail['etapes'][max(0, travail['etape'] - 1)]}\n\n"
            + "\n".join(lignes)
        ),
    }


@skill(
    description="Arrête le travail CalculRisque en cours.",
    examples=["arrête le pipeline", "stoppe l'optimisation", "annule la mise à jour"],
    confirm="Voulez-vous vraiment arrêter le travail en cours ?",
)
def arreter_le_travail() -> str:
    """Tue le sous-processus courant. Ce qui était écrit reste écrit."""
    with _VERROU:
        if not _TRAVAIL or _TRAVAIL.get("fini"):
            return "Il n'y a rien à arrêter."
        libelle = _TRAVAIL.get("libelle", "le travail")
        # Le drapeau AVANT le kill : entre « lance » et « arrête », le
        # sous-processus peut n'être pas encore né. Le fil le verra et
        # renoncera plutôt que de démarrer un travail qu'on vient d'annuler.
        _TRAVAIL["arret"] = True
        _TRAVAIL["fini"] = f"{libelle} : arrêté à votre demande."
        processus = _TRAVAIL.get("processus")
    if processus is not None and processus.poll() is None:
        processus.kill()
        try:
            processus.wait(timeout=5.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - un kill qui traîne
            get_logger().warning("Le processus de %s met du temps à mourir.", libelle)
    return f"J'ai arrêté {libelle}. Ce qui était déjà écrit sur le disque y reste."


def _rapprocher(demande: str, table: dict) -> str | None:
    """Retrouve une clé malgré ce que la transcription en aura fait.

    « seuil d'entrée » doit tomber sur ``seuil_entree`` : la voix ne dicte ni
    les tirets bas ni les accents.
    """
    simple = _reduire(demande)
    if not simple:
        return None
    for clef in table:
        propre = _reduire(str(clef))
        if propre == simple or (len(simple) >= 4 and simple in propre):
            return clef
    return None


# Ce que la voix intercale et que les clés de configuration n'ont pas.
# « seuil d'entrée » doit tomber sur `seuil_entree` : sans ce ménage, le « d »
# de l'élision suffit à faire manquer la correspondance.
_LIAISONS = {"d", "l", "de", "du", "des", "la", "le", "les", "un", "une"}


def _reduire(texte: str) -> str:
    mots = re.split(r"[^a-z0-9]+", _sans_accents(texte).lower())
    return "".join(mot for mot in mots if mot and mot not in _LIAISONS)


def _sans_accents(texte: str) -> str:
    import unicodedata

    return "".join(
        caractere for caractere in unicodedata.normalize("NFKD", texte)
        if not unicodedata.combining(caractere)
    )


# --- compétences : les optimisations ----------------------------------------

@skill(
    description="Lance une optimisation de paramètres de stratégie sur CalculRisque.",
    examples=[
        "optimise les stops",
        "lance l'optimisation du seuil d'entrée",
        "optimise le seuil de rebalancement",
        "optimise la fraction de convergence",
    ],
    timeout=30.0,
)
def optimiser(quoi: str, strategie: str = "", debut: str = "", fin: str = "") -> dict:
    """Lance l'optimiseur demandé en tâche de fond, résultats en CSV.

    Args:
        quoi: Ce qu'on optimise : stops, rebalancement, convergence, seuil_entree, multiples.
        strategie: La stratégie à optimiser. Vide, celle de la configuration.
        debut: Date de début du backtest, AAAA-MM-JJ. Vide, le défaut du script.
        fin: Date de fin, AAAA-MM-JJ. Vide, le défaut du script.
    """
    optimisations = dict(get_config("optimisations", {}) or {})
    clef = _rapprocher(quoi, optimisations)
    if clef is None:
        raise SkillRefused(
            f"Je ne sais pas optimiser « {quoi} ». "
            f"Je connais : {', '.join(sorted(optimisations))}."
        )
    script = str(optimisations[clef])

    dossier = str(get_config("dossier_optimisations", "data/optimisations"))
    horodatage = datetime.now().strftime("%Y%m%d-%H%M%S")
    sortie = f"{dossier}/{clef}_{horodatage}.csv"
    (_depot() / dossier).mkdir(parents=True, exist_ok=True)

    commande = [script]
    # `optimize_options_multiples.py` n'a ni --strategy ni --output-csv : ses
    # options ne sont pas celles des 11*. On ne lui passe que ce qu'il connaît.
    if "--output-csv" in _options_du_script(script):
        commande += ["--output-csv", sortie]
    else:
        sortie = ""
    if "--strategy" in _options_du_script(script):
        commande += ["--strategy", strategie.strip() or str(get_config("strategie_par_defaut", ""))]
    for drapeau, valeur in (("--start-date", debut), ("--end-date", fin)):
        if valeur.strip():
            commande += [drapeau, _date_valide(valeur)]

    ligne = " ".join(shlex.quote(morceau) if " " in morceau else morceau for morceau in commande)
    _lancer(f"optimisation {clef}", [ligne], sortie=Path(sortie) if sortie else None)
    return {
        "speak": f"C'est parti : optimisation {clef.replace('_', ' ')}. "
                 "Ça prend des heures ; je vous préviens à la fin.",
        "display": f"{ligne}\n\nRésultats attendus dans : {sortie or '(sortie standard)'}",
    }


def _options_du_script(script: str) -> set[str]:
    """Les drapeaux que ce script accepte réellement, lus dans sa source.

    Plutôt que de supposer : les cinq optimiseurs n'ont pas le même jeu
    d'options, et passer un drapeau inconnu fait échouer argparse avant même
    que le calcul commence.
    """
    chemin = _depot() / script
    if not chemin.is_file():
        raise SkillRefused(f"Le script « {script} » est absent du dépôt.")
    try:
        source = chemin.read_text(encoding="utf-8", errors="replace")
    except OSError as erreur:
        raise SkillRefused(f"Je ne peux pas lire « {script} » : {erreur}") from None
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', source))


def _date_valide(texte: str) -> str:
    texte = texte.strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", texte):
        raise SkillRefused(
            f"« {texte} » n'est pas une date au format année-mois-jour."
        )
    return texte


@skill(
    description="Lit le résultat de la dernière optimisation et donne les meilleurs réglages.",
    examples=[
        "qu'est-ce que l'optimisation a donné",
        "résultat de l'optimisation des stops",
        "montre-moi les meilleurs paramètres",
    ],
)
def resultat_optimisation(quoi: str = "", nombre: int = 5) -> dict:
    """Les meilleures lignes du CSV produit par un optimiseur.

    Args:
        quoi: Quelle optimisation. Vide, la plus récente toutes catégories.
        nombre: Combien de lignes montrer.
    """
    dossier = _depot() / str(get_config("dossier_optimisations", "data/optimisations"))
    if not dossier.is_dir():
        raise SkillRefused("Aucune optimisation n'a encore écrit de résultats.")
    motif = f"{_rapprocher(quoi, dict.fromkeys(_familles())) or quoi}_*.csv" if quoi.strip() else "*.csv"
    fichiers = sorted(dossier.glob(motif), key=lambda f: f.stat().st_mtime)
    if not fichiers:
        raise SkillRefused(f"Aucun résultat pour « {quoi or 'une optimisation'} ».")

    dernier = fichiers[-1]
    with dernier.open(encoding="utf-8", newline="") as fichier:
        lignes = list(csv.DictReader(fichier))
    if not lignes:
        raise SkillRefused(f"{dernier.name} est vide : l'optimisation a-t-elle abouti ?")

    # Les optimiseurs écrivent déjà trié par objectif décroissant : on ne
    # retrie pas sur une colonne qu'on aurait devinée.
    tete = lignes[: max(1, int(nombre))]
    colonnes = list(tete[0])
    tableau = [" | ".join(colonnes)]
    tableau += [" | ".join(str(ligne.get(colonne, ""))[:18] for colonne in colonnes) for ligne in tete]
    meilleur = ", ".join(f"{colonne} {tete[0][colonne]}" for colonne in colonnes[:3])
    return {
        "speak": f"D'après {dernier.name}, le meilleur réglage est : {meilleur}.",
        "display": f"{dernier}\n{len(lignes)} combinaisons testées\n\n" + "\n".join(tableau),
    }


def _familles() -> list[str]:
    return list(dict(get_config("optimisations", {}) or {}))


# --- compétences : les stratégies -------------------------------------------

GABARIT = '''"""Stratégie « {nom} », écrite par Capucine le {date}.

{intention}

Elle n'invente aucune mécanique : elle reprend celle de valuation_gap_dcf
(écart de valorisation corrigé de l'inflation, poids plafonnés par
capped_weights) avec les réglages dictés ci-dessous. Le stop-loss, le
take-profit et l'exécution restent au moteur — voir backtest/strategies/base.py.

C'est un fichier comme un autre : relisez-le, modifiez-le, supprimez-le.
"""

from __future__ import annotations

import pandas as pd

import config
from backtest.strategies.base import (
    Strategy,
    capped_weights,
    inflation_adjusted_gap,
    register_strategy,
)


@register_strategy("{registre}")
class {classe}(Strategy):
    """{resume}"""

    def __init__(
        self,
        entry_threshold_pct: float = {seuil},
        max_positions: int = {positions_max},
        **kwargs,
    ):
        super().__init__(
            entry_threshold_pct=entry_threshold_pct,
            max_positions=max_positions,
            **kwargs,
        )
        self.entry_threshold_pct = entry_threshold_pct
        self.max_positions = max_positions

    def generate_target_weights(
        self, signals: pd.DataFrame, current_positions: set[str]
    ) -> dict[str, float]:
        if signals.empty:
            return {{}}

        # Même correction d'inflation que les stratégies livrées : la valeur
        # théorique est nominale, la convergence se fait donc vers V x (1+pi)^T.
        signals = signals.assign(gap_pct=inflation_adjusted_gap(
            signals["gap_pct"], signals["published_date"],
            config.INFLATION_HORIZON_YEARS_STOCKS,
        ))
{filtre_secteur}
        conviction = {conviction}
        candidates = signals[conviction >= self.entry_threshold_pct]
        if candidates.empty:
            return {{}}
        conviction = conviction.loc[candidates.index]

        # Une conviction négative ferait diviser capped_weights par une somme
        # qui peut s'annuler : on la ramène dans le positif sans changer le
        # classement (cf. valuation_gap_sector_neutral).
        if (conviction <= 0).any():
            conviction = conviction - conviction.min() + 1e-9

        if self.max_positions > 0 and len(conviction) > self.max_positions:
            retenues = conviction.nlargest(self.max_positions).index
            candidates = candidates.loc[retenues]
            conviction = conviction.loc[retenues]

        weights = {ponderation}
        return dict(zip(candidates["symbol"], weights))
'''

_CONVICTIONS = {
    "ecart": 'signals["gap_pct"]',
    "excedent_sectoriel": (
        'signals["gap_pct"] - signals.groupby("sector")["gap_pct"].transform("median")'
    ),
}
_PONDERATIONS = {
    "ecart": "capped_weights(conviction)",
    "egale": "pd.Series(1.0 / len(candidates), index=candidates.index)",
}


@skill(
    description="Crée une nouvelle stratégie de backtest dans CalculRisque, à partir d'un gabarit.",
    examples=[
        "crée une nouvelle stratégie",
        "fais-moi une stratégie à trente pour cent d'écart",
        "ajoute une stratégie neutre par secteur",
    ],
    confirm="Je vais écrire un nouveau fichier de stratégie dans le dépôt. Je continue ?",
    timeout=600.0,
)
def creer_une_strategie(
    nom: str,
    seuil_entree: float = 25.0,
    conviction: str = "ecart",
    ponderation: str = "ecart",
    positions_max: int = 0,
    secteurs: str = "",
) -> dict:
    """Écrit la stratégie, l'enregistre, puis prouve qu'elle charge.

    Args:
        nom: Le nom de la stratégie, par exemple « écart profond ».
        seuil_entree: Écart de valorisation minimum, en pour cent.
        conviction: Sur quoi classer : « ecart » brut, ou « excedent_sectoriel »
            (l'écart moins la médiane du secteur).
        ponderation: « ecart » pour des poids proportionnels plafonnés,
            « egale » pour l'équipondération.
        positions_max: Nombre maximum de positions. Zéro : sans limite.
        secteurs: Secteurs à ne garder, séparés par des virgules. Vide : tous.
    """
    depot = _depot()
    registre = _identifiant(nom)
    if not registre:
        raise SkillRefused("Ce nom ne donne aucun identifiant utilisable.")
    if conviction not in _CONVICTIONS:
        raise SkillRefused(
            f"« {conviction} » : je connais {', '.join(_CONVICTIONS)}."
        )
    if ponderation not in _PONDERATIONS:
        raise SkillRefused(
            f"« {ponderation} » : je connais {', '.join(_PONDERATIONS)}."
        )

    relatif = f"backtest/strategies/{registre}.py"
    if (depot / relatif).exists():
        raise SkillRefused(
            f"« {registre} » existe déjà. Choisissez un autre nom : je n'écrase "
            "jamais une stratégie existante."
        )

    classe = "".join(morceau.capitalize() for morceau in registre.split("_")) + "Strategy"
    code = GABARIT.format(
        nom=nom.strip(), date=datetime.now().date().isoformat(),
        intention=_plier(_intention(
            seuil_entree, conviction, ponderation, positions_max, secteurs)),
        registre=registre, classe=classe,
        resume=f"Écart minimum {float(seuil_entree):g} %, conviction {conviction}, pondération {ponderation}.",
        seuil=float(seuil_entree), positions_max=max(0, int(positions_max)),
        filtre_secteur=_filtre_secteur(secteurs),
        conviction=_CONVICTIONS[conviction],
        ponderation=_PONDERATIONS[ponderation],
    )
    compile(code, relatif, "exec")      # jamais de fichier non compilable déposé

    # L'atelier écrit : c'est lui qui garde une sauvegarde et qui vérifie que
    # le chemin reste dans le périmètre autorisé.
    atelier().ecrire(depot / relatif, code)
    _enregistrer(depot, registre, classe)

    verdict = _verifier(depot, registre)
    get_logger().info("Stratégie créée : %s (%s)", registre, verdict)
    return {
        "speak": f"La stratégie {nom.strip()} est écrite et enregistrée. {verdict}",
        "display": f"{depot / relatif}\n"
                   f"nom dans le registre : {registre}\n"
                   f"classe : {classe}\n\n{verdict}\n\n"
                   f"Pour la jouer : python 09_backtest.py --strategy {registre}",
    }


def _identifiant(nom: str) -> str:
    propre = re.sub(r"[^a-z0-9]+", "_", _sans_accents(nom).lower()).strip("_")
    if propre and propre[0].isdigit():
        propre = f"s_{propre}"
    return propre[:40]


def _plier(texte: str, largeur: int = 76) -> str:
    """Plie l'intention à la largeur du dépôt : son code est plié, pas le mien."""
    import textwrap

    return "\n".join(textwrap.wrap(texte, largeur)) or texte


def _intention(seuil, conviction, ponderation, positions_max, secteurs) -> str:
    morceaux = [
        f"Achète les entreprises dont l'écart de valorisation atteint {float(seuil):g} %"
    ]
    if conviction == "excedent_sectoriel":
        morceaux.append("mesuré en excédent sur la médiane de leur secteur")
    if secteurs.strip():
        morceaux.append(f"parmi les secteurs {secteurs.strip()}")
    if int(positions_max) > 0:
        morceaux.append(f"au plus {int(positions_max)} lignes à la fois")
    morceaux.append(
        "pondérées par l'ampleur de l'écart" if ponderation == "ecart"
        else "à parts égales"
    )
    return ", ".join(morceaux) + "."


def _filtre_secteur(secteurs: str) -> str:
    noms = [morceau.strip() for morceau in secteurs.split(",") if morceau.strip()]
    if not noms:
        return ""
    liste = ", ".join(repr(nom) for nom in noms)
    return (
        f"\n        signals = signals[signals[\"sector\"].isin([{liste}])]\n"
        "        if signals.empty:\n"
        "            return {}\n"
    )


def _enregistrer(depot: Path, registre: str, classe: str) -> None:
    """Ajoute la ligne d'import qui met la stratégie dans le registre.

    C'est la SEULE modification d'un fichier existant que ce plugin s'autorise,
    et l'atelier en garde une sauvegarde. Idempotente : réécrire une ligne déjà
    présente ne la duplique pas.
    """
    chemin = depot / "backtest" / "strategies" / "__init__.py"
    source = chemin.read_text(encoding="utf-8")
    ligne = f"from backtest.strategies.{registre} import {classe}"
    if ligne in source:
        return
    marqueur = "\n__all__ = ["
    if marqueur not in source:
        raise SkillRefused(
            "backtest/strategies/__init__.py n'a pas la forme attendue "
            "(pas de __all__). Je préfère ne pas y toucher à l'aveugle ; "
            f"ajoutez vous-même : {ligne}"
        )
    source = source.replace(marqueur, f"\n{ligne}\n{marqueur}", 1)
    source = source.replace('__all__ = [', f'__all__ = [\n    "{classe}",', 1)
    atelier().ecrire(chemin, source)


def _verifier(depot: Path, registre: str) -> str:
    """Prouve que la stratégie charge, en interrogeant le vrai registre.

    Écrire un fichier qui ne s'importe pas serait pire que ne rien écrire :
    la panne n'apparaîtrait qu'au prochain backtest, sans lien apparent.
    """
    try:
        resultat = subprocess.run(
            [sys.executable, "09_backtest.py", "--list-strategies"],
            cwd=str(depot), env=_environnement(), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as erreur:
        return f"⚠ Je n'ai pas pu vérifier qu'elle charge : {erreur}"
    if registre in resultat.stdout.split():
        return "Je l'ai vérifiée : le registre la reconnaît."
    detail = (resultat.stderr or resultat.stdout).strip().splitlines()
    return (
        "⚠ Le fichier est écrit, mais le registre ne la voit pas encore. "
        + (detail[-1] if detail else "")
    )


@skill(
    description="Dit quelles stratégies de backtest le projet CalculRisque connaît.",
    examples=[
        "quelles stratégies tu connais",
        "liste les stratégies du projet",
        "quelles stratégies d'options existent",
    ],
    timeout=300.0,
)
def mes_strategies() -> dict:
    """Interroge les deux registres — actions et options — dans le vrai dépôt."""
    depot = _depot()
    resultats = {}
    for libelle, script in (("actions", "09_backtest.py"), ("options", "10_backtest_options.py")):
        try:
            sortie = subprocess.run(
                [sys.executable, script, "--list-strategies"],
                cwd=str(depot), env=_environnement(), capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=180,
            )
        except (OSError, subprocess.SubprocessError) as erreur:
            resultats[libelle] = [f"(illisible : {erreur})"]
            continue
        resultats[libelle] = sortie.stdout.split() or [f"(aucune — {script} a rendu {sortie.returncode})"]

    total = sum(len(noms) for noms in resultats.values())
    return {
        "speak": f"Le projet connaît {total} stratégies : "
                 + ", ".join(f"{len(noms)} pour les {libelle}"
                             for libelle, noms in resultats.items()) + ".",
        "display": "\n\n".join(
            f"{libelle} :\n" + "\n".join(f"  {nom}" for nom in noms)
            for libelle, noms in resultats.items()
        ),
    }


@skill(
    description="Joue un backtest d'une stratégie du projet CalculRisque.",
    examples=[
        "backteste la stratégie valuation gap dcf",
        "joue le backtest de ma nouvelle stratégie",
        "lance un backtest sur deux mille vingt",
    ],
    timeout=30.0,
)
def backtester(strategie: str, debut: str = "", fin: str = "", options: bool = False) -> str:
    """Lance 09 (actions) ou 10 (options) en tâche de fond.

    Args:
        strategie: Le nom de la stratégie dans le registre.
        debut: Date de début, AAAA-MM-JJ. Vide, le défaut du script.
        fin: Date de fin, AAAA-MM-JJ. Vide, le défaut du script.
        options: Vrai pour le moteur d'options (10) au lieu des actions (09).
    """
    nom = strategie.strip().replace(" ", "_")
    if not nom:
        raise SkillRefused("Dites-moi quelle stratégie backtester.")
    script = "10_backtest_options.py" if options else "09_backtest.py"
    commande = [script, "--strategy", nom]
    for drapeau, valeur in (("--start-date", debut), ("--end-date", fin)):
        if valeur.strip():
            commande += [drapeau, _date_valide(valeur)]

    _lancer(f"backtest {nom}", [" ".join(commande)])
    return (
        f"Backtest de {nom} lancé sur le moteur "
        f"{'options' if options else 'actions'}. Je vous préviens à la fin."
    )

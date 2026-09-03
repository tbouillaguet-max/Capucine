"""Lancer un projet entier, et suivre son avancement à la voix.

Ce qu'il montre :

* une **tâche de fond longue**. Un pipeline de données tourne quarante
  minutes : hors de question de bloquer un tour de parole. On lance, on rend
  la main tout de suite, et ``announce()`` interrompt quand c'est fini ;
* la **configuration comme donnée**. Les projets sont déclarés dans le fichier
  de configuration, pas dans le code : ajouter un projet ne demande pas de
  toucher au plugin ;
* la réutilisation de l'**atelier** comme périmètre de sécurité. Un projet
  doit se trouver dans un dossier que vous avez ouvert, sans quoi Lily
  refuse de le lancer.

Déclaration d'un projet, dans ``config/pc.toml`` ::

    [plugins.projet.projets.calculrisque]
    chemin = "~/projets/CalculRisque_Mark5"
    description = "pipeline de valorisation et backtest options"
    commande = "run_pipeline_quarterly.py"
    options_par_defaut = "--resume"
    rapport = "data/pipeline_runs/*/report.json"
    commande_test = "-m pytest -q"
    delai_s = 5400
    variables = { SEC_CONTACT_EMAIL = "vous@exemple.fr" }
"""

import glob
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from lily.plugin import SkillRefused, announce, atelier, get_config, get_logger, skill

CONFIG_DEFAULTS = {
    "projets": {},
    "delai_s": 3600.0,
    "lignes_journal": 25,
}

# Un run en cours par projet : processus, départ, journal.
_EN_COURS: dict[str, dict] = {}
_VERROU = threading.Lock()


def on_unload() -> None:
    """Arrête les runs en cours avant que le module ne disparaisse.

    Sans cela, un rechargement à chaud laisserait des sous-processus orphelins
    dont plus personne ne lirait la sortie.
    """
    with _VERROU:
        for nom, run in list(_EN_COURS.items()):
            processus = run.get("processus")
            if processus and processus.poll() is None:
                get_logger().info("Arrêt du projet %s au déchargement.", nom)
                processus.kill()
        _EN_COURS.clear()


def _projets() -> dict:
    return dict(get_config("projets", {}) or {})


def _definition(nom: str) -> tuple[str, dict]:
    projets = _projets()
    if not projets:
        raise SkillRefused(
            "Aucun projet n'est déclaré. Ajoutez une section "
            "[plugins.projet.projets.<nom>] dans votre configuration."
        )
    nom = (nom or "").strip().lower()
    if not nom and len(projets) == 1:
        return next(iter(projets.items()))
    for cle, definition in projets.items():
        if nom in cle.lower() or nom in str(definition.get("description", "")).lower():
            return cle, definition
    raise SkillRefused(
        f"Je ne connais pas de projet « {nom} ». J'ai : {', '.join(projets)}."
    )


def _dossier(definition: dict) -> Path:
    chemin = definition.get("chemin")
    if not chemin:
        raise SkillRefused("Ce projet n'a pas de chemin dans la configuration.")
    # Passe par l'atelier : un projet hors des dossiers ouverts est refusé.
    return atelier().resoudre(chemin, doit_exister=True)


def _environnement(definition: dict) -> dict:
    variables = os.environ.copy()
    variables.update({str(k): str(v) for k, v in (definition.get("variables") or {}).items()})
    return variables


def _suivre(nom: str, definition: dict, options: str) -> None:
    """Corps du thread : lance, attend, annonce. Ne lève jamais."""
    dossier = _dossier(definition)
    # shlex plutôt que split() : « -k "not slow" » doit rester un seul argument.
    commande = shlex.split(str(definition.get("commande", "")))
    if not commande:
        announce(f"Le projet {nom} n'a pas de commande configurée.")
        return
    interpreteur = str(definition.get("interpreteur", "")) or sys.executable
    arguments = [interpreteur, *commande, *shlex.split(options)]
    delai = float(definition.get("delai_s", get_config("delai_s", 3600.0)))

    journal: list[str] = []
    depart = time.monotonic()
    get_logger().info("Projet %s : %s", nom, " ".join(arguments))
    try:
        processus = subprocess.Popen(  # noqa: S603 - jamais de shell
            arguments, cwd=str(dossier), env=_environnement(definition),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
    except OSError as exc:
        announce(f"Je n'ai pas pu lancer {nom} : {exc}.")
        return

    with _VERROU:
        _EN_COURS[nom] = {"processus": processus, "depart": depart, "journal": journal}

    try:
        for ligne in processus.stdout or []:
            journal.append(ligne.rstrip())
            del journal[:-400]      # on ne garde que la fin
            if time.monotonic() - depart > delai:
                processus.kill()
                announce(f"{nom} tournait encore après le délai prévu, je l'ai arrêté.")
                return
        code = processus.wait()
    except Exception:  # pragma: no cover - un tuyau qui casse ne tue rien
        get_logger().exception("Suivi du projet %s interrompu.", nom)
        code = -1
    finally:
        with _VERROU:
            _EN_COURS.pop(nom, None)

    minutes = (time.monotonic() - depart) / 60
    if code == 0:
        announce(f"{nom} est terminé, en {minutes:.0f} minutes.")
    else:
        derniere = next((ligne for ligne in reversed(journal) if ligne.strip()), "sans message")
        announce(f"{nom} s'est arrêté en erreur après {minutes:.0f} minutes : {derniere[:160]}")


@skill(
    description="Lance un projet en tâche de fond et prévient quand il a fini.",
    examples=[
        "lance le pipeline calculrisque",
        "fais tourner le projet",
        "démarre le backtest complet",
    ],
    confirm="Voulez-vous vraiment lancer ce projet ? Cela peut durer longtemps.",
)
def lancer_projet(nom: str = "", options: str = "") -> str:
    """Démarre un projet configuré, sans attendre sa fin.

    Args:
        nom: Le projet à lancer. Vide s'il n'y en a qu'un.
        options: Options de ligne de commande à ajouter.
    """
    cle, definition = _definition(nom)
    with _VERROU:
        if cle in _EN_COURS:
            return f"{cle} tourne déjà. Demandez-moi où il en est."

    options = options.strip() or str(definition.get("options_par_defaut", ""))
    fil = threading.Thread(
        target=_suivre, args=(cle, definition, options), name=f"projet-{cle}", daemon=True
    )
    fil.start()
    time.sleep(0.2)   # laisse le temps au processus de démarrer ou d'échouer

    description = definition.get("description", "")
    return f"C'est parti pour {cle}{', ' + description if description else ''}. Je vous préviens à la fin."


@skill(
    description="Dit où en est un projet en cours d'exécution.",
    examples=["où en est le pipeline", "ça avance", "l'état du projet"],
)
def etat_projet(nom: str = "") -> dict:
    """Donne l'avancement d'un run en cours, ou le résultat du dernier.

    Args:
        nom: Le projet concerné. Vide s'il n'y en a qu'un.
    """
    cle, definition = _definition(nom)
    with _VERROU:
        run = _EN_COURS.get(cle)
        if run is not None:
            minutes = (time.monotonic() - run["depart"]) / 60
            journal = list(run["journal"])

    if run is not None:
        derniere = next((ligne for ligne in reversed(journal) if ligne.strip()), "")
        lignes = int(get_config("lignes_journal", 25))
        return {
            "speak": f"{cle} tourne depuis {minutes:.0f} minutes. {derniere[:200]}",
            "display": "\n".join(journal[-lignes:]) or "(pas encore de sortie)",
        }

    rapport = _dernier_rapport(definition)
    if rapport is None:
        return {"speak": f"{cle} ne tourne pas.", "display": f"{cle} : à l'arrêt"}

    statut = rapport.get("status", "inconnu")
    duree = rapport.get("duration_seconds", 0) / 60
    etapes = rapport.get("steps", [])
    echecs = [e.get("script") for e in etapes if e.get("status") == "failed"]
    parle = f"Le dernier run de {cle} s'est terminé en {statut}, en {duree:.0f} minutes."
    if echecs:
        parle += f" Étapes en échec : {', '.join(echecs)}."
    return {
        "speak": parle,
        "display": json.dumps(
            {"statut": statut, "duree_min": round(duree, 1),
             "etapes": [(e.get("script"), e.get("status")) for e in etapes]},
            ensure_ascii=False, indent=2,
        ),
    }


def _dernier_rapport(definition: dict) -> dict | None:
    """Lit le rapport JSON du dernier run, si le projet en produit un."""
    motif = definition.get("rapport")
    if not motif:
        return None
    try:
        dossier = _dossier(definition)
    except SkillRefused:
        return None
    candidats = sorted(glob.glob(str(dossier / motif)))
    if not candidats:
        return None
    try:
        return json.loads(Path(candidats[-1]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@skill(
    description="Arrête un projet en cours d'exécution.",
    examples=["arrête le pipeline", "stoppe le projet", "annule le run"],
    confirm="Voulez-vous vraiment arrêter ce projet en cours ?",
)
def arreter_projet(nom: str = "") -> str:
    """Tue le processus d'un run en cours.

    Args:
        nom: Le projet à arrêter. Vide s'il n'y en a qu'un.
    """
    cle, _ = _definition(nom)
    with _VERROU:
        run = _EN_COURS.get(cle)
        if run is None:
            return f"{cle} ne tourne pas."
        run["processus"].kill()
    return f"{cle} arrêté."


@skill(
    description="Lance la suite de tests d'un projet.",
    examples=["lance les tests du projet", "fais tourner les tests", "vérifie que tout passe"],
    confirm="Voulez-vous vraiment lancer les tests ?",
    timeout=900.0,
)
def tester_projet(nom: str = "") -> dict:
    """Exécute la commande de test configurée, et attend le résultat.

    Args:
        nom: Le projet à tester. Vide s'il n'y en a qu'un.
    """
    cle, definition = _definition(nom)
    commande = shlex.split(str(definition.get("commande_test", "")))
    if not commande:
        raise SkillRefused(f"{cle} n'a pas de commande de test configurée.")

    interpreteur = str(definition.get("interpreteur", "")) or sys.executable
    try:
        resultat = subprocess.run(  # noqa: S603 - jamais de shell
            [interpreteur, *commande], cwd=str(_dossier(definition)),
            env=_environnement(definition), capture_output=True, text=True,
            timeout=float(definition.get("delai_test_s", 600)), check=False,
        )
    except subprocess.TimeoutExpired:
        raise SkillRefused("Les tests tournaient encore après le délai, je les ai arrêtés.") from None

    sortie = ((resultat.stdout or "") + (resultat.stderr or "")).strip().splitlines()
    derniere = next((ligne for ligne in reversed(sortie) if ligne.strip()), "sans message")
    verdict = "Les tests passent." if resultat.returncode == 0 else "Les tests échouent."
    return {"speak": f"{verdict} {derniere[:200]}", "display": "\n".join(sortie[-30:])}


@skill(
    description="Liste les projets que Lily sait lancer.",
    examples=["quels projets connais-tu", "mes projets", "que peux-tu lancer"],
)
def mes_projets() -> dict:
    """Les projets déclarés dans la configuration."""
    projets = _projets()
    if not projets:
        return {
            "speak": "Aucun projet n'est déclaré pour l'instant.",
            "display": "ajoutez [plugins.projet.projets.<nom>] dans la configuration",
        }
    with _VERROU:
        actifs = set(_EN_COURS)
    morceaux = [
        f"{cle}{' (en cours)' if cle in actifs else ''}" for cle in projets
    ]
    return {
        "speak": "Je connais " + ", ".join(morceaux) + ".",
        "display": "\n".join(
            f"{cle} — {d.get('description', 'sans description')} — {d.get('chemin', '?')}"
            for cle, d in projets.items()
        ),
    }

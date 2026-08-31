#!/usr/bin/env python3
"""Banc de mesure : lequel des trois leviers fait vraiment coder Capucine ?

    python tools/banc_de_code.py --atelier ~/projets/CalculRisque_Mark5
    python tools/banc_de_code.py --atelier … --variantes          # les quatre combinaisons
    python tools/banc_de_code.py --atelier … --llm ollama --modele qwen2.5-coder:7b

Sans lui, choisir entre « changer de modèle », « donner l'API en contexte » et
« boucler sur l'erreur » est un acte de foi. Avec lui, ça se lit.

**La notation ne juge rien.** Chaque tâche déclare une vérification en Python ;
le code produit et sa vérification sont exécutés ensemble, et la tâche passe si
le processus rend zéro. Pas de modèle-juge, pas d'appréciation : la même
récompense vérifiable qui a fait progresser les modèles de code.

Les tâches vivent dans ``tools/banc/taches.toml``. Ajoutez les vôtres — un banc
n'a de valeur que s'il ressemble à ce que vous demandez vraiment.

Ce que le banc NE mesure PAS : si le résultat est juste au sens métier. Une
fonction qui tourne et rend un mauvais chiffre passe. C'est une mesure de
« ça marche », pas de « c'est bon ».
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

TACHES = Path(__file__).resolve().parent / "banc" / "taches.toml"


@dataclass
class Tache:
    nom: str
    description: str
    verification: str
    famille: str = "logique"
    doit_appeler: list[str] = field(default_factory=list)


@dataclass
class Resultat:
    tache: str
    famille: str
    reussi: bool
    raison: str
    secondes: float
    lignes: int
    code: str = ""
    entrees_api: int = 0      # signatures réellement injectées dans le contexte
    caracteres_api: int = 0


def charger_les_taches(chemin: Path) -> list[Tache]:
    donnees = tomllib.loads(chemin.read_text(encoding="utf-8"))
    return [Tache(**entree) for entree in donnees.get("tache", [])]


# --- montage minimal de Capucine -------------------------------------------

def monter(
    atelier_chemin: Path, llm: str, modele: str, catalogue_actif: bool,
    delai_modele: float = 300.0, delai_competence: float = 600.0,
):
    """Un assistant réduit : le registre, l'atelier, le catalogue, le modèle.

    On passe par les VRAIES compétences (`ecrire_du_code`, `coder_et_verifier`)
    plutôt que de réimplémenter la boucle ici — sans quoi le banc mesurerait
    autre chose que ce que Capucine fait.
    """
    from capucine.core import plugin as contrat
    from capucine.core.atelier import depuis_config as atelier_depuis_config
    from capucine.core.catalogue import Catalogue
    from capucine.core.config import PROJECT_ROOT, load_config
    from capucine.core.engines.factory import build_llm
    from capucine.core.interfaces.llm import Message
    from capucine.core.registry import PluginRegistry

    # Un banc n'est pas un tour de parole : les délais taillés pour la voix
    # (60 s côté client Ollama, 120 s côté compétence) coupent une génération
    # de 900 jetons sur un 7B en CPU, et la coupure ressort en « échec » alors
    # que rien n'a échoué. On mesure le modèle, pas la patience.
    surcharges = {
        "atelier": {"racines": [str(atelier_chemin)],
                    "corbeille": str(atelier_chemin / ".corbeille_banc")},
        "plugins": {"python": {"api_en_contexte": catalogue_actif},
                    "timeout": delai_competence},
        "llm": {"timeout": delai_modele},
    }
    if llm:
        surcharges["llm"]["engine"] = llm
    if modele:
        surcharges["llm"]["model"] = modele
    config = load_config(overrides=surcharges)

    moteur = build_llm(config)

    def demander(prompt, *, system="", max_tokens=512, temperature=0.2, json_schema=None):
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        return moteur.chat(messages, json_schema=json_schema,
                           temperature=temperature, max_tokens=max_tokens)

    contrat.set_atelier(atelier_depuis_config(config))
    contrat.set_catalogue(Catalogue(atelier_chemin) if catalogue_actif else None)
    contrat.set_model_access(demander)
    registre = PluginRegistry(
        [PROJECT_ROOT / "plugins"], config=config,
        data_root=atelier_chemin / ".banc_data",
    )
    registre.load_all()
    # `@skill(timeout=...)` est écrit en dur dans le plugin et l'emporte sur
    # `plugins.timeout` : on le relève ici, explicitement, pour ces deux
    # compétences seulement.
    for nom in ("ecrire_du_code", "coder_et_verifier"):
        specification = registre.get(nom)
        if specification is not None:
            specification.timeout = delai_competence
    return registre, moteur


def demonter(registre) -> None:
    from capucine.core import plugin as contrat

    for nom in list(registre.plugins):
        registre.unload(nom, notify=False)
    contrat.set_atelier(None)
    contrat.set_catalogue(None)
    contrat.set_model_access(None)


# --- la notation ------------------------------------------------------------

def noter(code: str, tache: Tache, depot: Path, delai: float) -> tuple[bool, str]:
    """Lance le code produit ET la vérification. Zéro, c'est réussi."""
    if not code.strip():
        return False, "aucun code produit"
    manquantes = [nom for nom in tache.doit_appeler if nom not in code]
    if manquantes:
        # Le cœur de la mesure du catalogue : le modèle a-t-il appelé LA
        # fonction du projet, ou en a-t-il inventé une qui sonnait juste ?
        return False, f"n'appelle pas {', '.join(manquantes)}"

    programme = f"{code}\n\n# --- vérification ---\n{tache.verification}\n"
    try:
        resultat = subprocess.run(
            [sys.executable, "-c", programme],
            cwd=str(depot), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=delai, check=False,
        )
    except (OSError, subprocess.SubprocessError) as erreur:
        return False, f"impossible à lancer : {erreur}"
    if resultat.returncode == 0:
        return True, "vérification passée"
    derniere = (resultat.stderr or "").strip().splitlines()
    return False, derniere[-1][:120] if derniere else f"code de retour {resultat.returncode}"


def passer_une_tache(registre, tache: Tache, depot: Path, boucle: bool, delai: float) -> Resultat:
    depart = time.perf_counter()
    competence = "coder_et_verifier" if boucle else "ecrire_du_code"
    try:
        sortie = registre.call(competence, {"description": tache.description}, confirmed=True)
    except Exception as erreur:  # pragma: no cover - une panne de moteur
        return Resultat(tache.nom, tache.famille, False, f"compétence en erreur : {erreur}",
                        time.perf_counter() - depart, 0)

    from capucine.core import plugin as contrat

    code = ""
    api = ""
    module = sys.modules.get("capucine.plugins.python")
    if module is not None:
        proposition = getattr(module, "_PROPOSITION", {})
        code = proposition.get("code", "")
        # Ce que le modèle a VRAIMENT reçu — pas ce qu'on croit lui avoir
        # donné. Sans cette colonne, un temps qui explose sur une tâche
        # censée ne rien recevoir reste une énigme.
        api = proposition.get("api", "")
    del contrat
    # Une entrée commence par sa ligne d'import — ou par le commentaire qui
    # dit pourquoi il n'y en a pas. Compter « def » ne marche plus depuis que
    # la référence est rendue en Python valide, et rendait la colonne muette.
    entrees_api = sum(
        1 for ligne in api.splitlines()
        if ligne.startswith("from ") or ligne.startswith("# défini dans")
    )
    reussi, raison = noter(code, tache, depot, delai)
    if not sortie.ok and not reussi:
        # Un délai dépassé n'est pas un échec du modèle : le distinguer évite
        # de conclure qu'une configuration est mauvaise alors qu'elle est
        # seulement lente.
        message = sortie.speak
        raison = ("⏱ délai dépassé — relancez avec --delai-competence plus haut"
                  if "trop de temps" in message else message[:120])
    return Resultat(tache.nom, tache.famille, reussi, raison,
                    time.perf_counter() - depart, len(code.splitlines()),
                    code, entrees_api, len(api))


# --- restitution ------------------------------------------------------------

def ligne_de(resultat: Resultat) -> str:
    marque = "✓" if resultat.reussi else "✗"
    api = f"{resultat.entrees_api}fn/{resultat.caracteres_api}c" if resultat.caracteres_api else "—"
    return (
        f"  {marque} {resultat.tache:<18} {resultat.famille:<8} "
        f"{resultat.secondes:6.1f}s {resultat.lignes:3d}l {api:>10}  {resultat.raison[:44]}"
    )


def pied(resultats: list[Resultat]) -> str:
    reussis = sum(1 for resultat in resultats if resultat.reussi)
    duree = sum(resultat.secondes for resultat in resultats)
    return f"  → {reussis}/{len(resultats)} réussies en {duree:.0f} s"


def resume(mesures: dict[str, list[Resultat]]) -> str:
    """Le tableau qui répond à la question : lequel des leviers paye ?"""
    familles = sorted({resultat.famille for lot in mesures.values() for resultat in lot})
    entete = f"{'configuration':<28} {'total':>7}  " + "  ".join(f"{f:>8}" for f in familles)
    lignes = ["", entete, "─" * len(entete)]
    for nom, lot in mesures.items():
        reussis = sum(1 for resultat in lot if resultat.reussi)
        cellules = []
        for famille in familles:
            sous = [resultat for resultat in lot if resultat.famille == famille]
            cellules.append(f"{sum(1 for r in sous if r.reussi)}/{len(sous):<6}")
        lignes.append(f"{nom:<28} {reussis:>3}/{len(lot):<3}  " + "  ".join(f"{c:>8}" for c in cellules))
    return "\n".join(lignes)


# --- entrée -----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="banc_de_code", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--atelier", type=Path, required=True,
                        help="le dépôt sur lequel on mesure (et dont on lit l'API)")
    parser.add_argument("--taches", type=Path, default=TACHES)
    parser.add_argument("--llm", default="", help="ollama, llamacpp, mock…")
    parser.add_argument("--modele", default="", help="par exemple qwen2.5-coder:7b")
    parser.add_argument("--sans-catalogue", action="store_true")
    parser.add_argument("--sans-boucle", action="store_true")
    parser.add_argument("--variantes", action="store_true",
                        help="mesure les quatre combinaisons catalogue × boucle")
    parser.add_argument("--delai", type=float, default=60.0,
                        help="délai d'exécution du code produit (s)")
    parser.add_argument("--delai-modele", type=float, default=300.0,
                        help="délai de lecture côté Ollama (s)")
    parser.add_argument("--delai-competence", type=float, default=600.0,
                        help="délai d'une compétence du banc (s)")
    parser.add_argument("--montrer-code", action="store_true",
                        help="affiche le code produit par les tâches ratées")
    parser.add_argument("--verbeux", action="store_true",
                        help="garde les traces du registre (bruyantes)")
    parser.add_argument("--json", type=Path, help="écrit les résultats bruts")
    args = parser.parse_args(argv)

    from capucine.core.logging import setup_logging

    setup_logging(level="WARNING")
    if not args.verbeux:
        # Le banc rapporte lui-même chaque échec, avec sa raison. La trace
        # complète du registre par-dessus noierait le tableau.
        import logging

        logging.getLogger("capucine.registre").setLevel(logging.CRITICAL)

    depot = args.atelier.expanduser().resolve()
    if not depot.is_dir():
        print(f"Dépôt introuvable : {depot}", file=sys.stderr)
        return 2
    taches = charger_les_taches(args.taches)
    if not taches:
        print(f"Aucune tâche dans {args.taches}", file=sys.stderr)
        return 2

    if args.variantes:
        combinaisons = [
            ("sans rien", False, False),
            ("catalogue seul", True, False),
            ("boucle seule", False, True),
            ("catalogue + boucle", True, True),
        ]
    else:
        combinaisons = [(
            "catalogue " + ("off" if args.sans_catalogue else "on")
            + ", boucle " + ("off" if args.sans_boucle else "on"),
            not args.sans_catalogue, not args.sans_boucle,
        )]

    mesures: dict[str, list[Resultat]] = {}
    for titre, catalogue_actif, boucle in combinaisons:
        registre, moteur = monter(
            depot, args.llm, args.modele, catalogue_actif,
            args.delai_modele, args.delai_competence,
        )
        try:
            if titre == combinaisons[0][0]:
                print(f"modèle : {moteur.describe()}   dépôt : {depot}")
                print(
                    f"{len(taches)} tâches × {len(combinaisons)} configuration(s) — "
                    "comptez plusieurs minutes par tâche sur un 7B en CPU.\n"
                )
            print(f"── {titre} " + "─" * max(0, 58 - len(titre)))
            lot = []
            for tache in taches:
                # Au fil de l'eau : une minute de silence par tâche rend le
                # banc indistinguable d'un blocage.
                resultat = passer_une_tache(
                    registre, tache, depot, boucle, args.delai
                )
                lot.append(resultat)
                print(ligne_de(resultat), flush=True)
                if args.montrer_code and not resultat.reussi and resultat.code:
                    # Après trois corrections à l'aveugle, la seule chose qui
                    # tranche est ce que le modèle a RÉELLEMENT écrit.
                    extrait = "\n".join(
                        f"      │ {ligne}" for ligne in resultat.code.splitlines()[:25]
                    )
                    print(extrait, flush=True)
        finally:
            demonter(registre)
        mesures[titre] = lot
        print(pied(lot))
        print()

    if len(mesures) > 1:
        print(resume(mesures))
        print(
            "\nLisez les colonnes, pas seulement le total : le catalogue se voit "
            "sur « api »,\nla boucle sur « logique ». Un levier qui ne bouge pas "
            "sa colonne ne paye pas."
        )
    if args.json:
        args.json.write_text(json.dumps(
            {titre: [vars(resultat) for resultat in lot] for titre, lot in mesures.items()},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")
        print(f"\nRésultats bruts : {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Piloter CalculRisque_Mark5 : enchaînements, optimisations, stratégies.

Aucun test ne touche au vrai dépôt de l'utilisateur : on construit un dépôt
factice qui a la même FORME (les drapeaux réels des scripts, un
backtest/strategies/__init__.py de même structure), et c'est cette forme que
le plugin exploite.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lily.core import plugin as contrat
from lily.core.atelier import depuis_config as atelier_depuis_config
from lily.core.config import PROJECT_ROOT, Config
from lily.core.registry import PluginRegistry

PLUGINS = PROJECT_ROOT / "plugins"

# Les drapeaux réels de 11d, lus dans le vrai dépôt : le plugin lit la source
# d'un script pour savoir ce qu'il accepte, plutôt que de le supposer.
OPTIMISEUR = '''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", default="")
parser.add_argument("--start-date", default=None)
parser.add_argument("--end-date", default=None)
parser.add_argument("--output-csv", default=None)
args = parser.parse_args()
print("optimiseur factice", args)
'''

# `optimize_options_multiples.py` n'a ni --strategy ni --output-csv.
OPTIMISEUR_NU = '''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=10)
args = parser.parse_args()
print("optimiseur nu")
'''

INIT_STRATEGIES = '''"""Registre des stratégies."""

from backtest.strategies.base import STRATEGY_REGISTRY, Strategy, register_strategy
from backtest.strategies.valuation_gap import ValuationGapDCFStrategy

__all__ = [
    "STRATEGY_REGISTRY", "Strategy", "register_strategy", "ValuationGapDCFStrategy",
]
'''

LISTE_STRATEGIES = '''
import sys
if "--list-strategies" in sys.argv:
    import re, pathlib
    source = pathlib.Path("backtest/strategies/__init__.py").read_text(encoding="utf-8")
    for nom in re.findall(r"from backtest\\.strategies\\.(\\w+) import", source):
        if nom not in ("base", "options_base"):
            print(nom)
'''

ETAPE_OK = "print('étape faite')\n"
ETAPE_KO = "import sys\nprint('je rate', flush=True)\nsys.exit(3)\n"
ETAPE_LENTE = "import time\nprint('je prends mon temps', flush=True)\ntime.sleep(30)\n"


@pytest.fixture
def depot(tmp_path: Path) -> Path:
    """Un dépôt factice à la forme du vrai."""
    racine = tmp_path / "CalculRisque_Mark5"
    (racine / "backtest" / "strategies").mkdir(parents=True)
    (racine / "backtest" / "strategies" / "__init__.py").write_text(
        INIT_STRATEGIES, encoding="utf-8")
    (racine / "backtest" / "strategies" / "base.py").write_text("", encoding="utf-8")
    (racine / "backtest" / "strategies" / "valuation_gap.py").write_text("", encoding="utf-8")
    for nom, code in (
        ("11_optimize_options_stops.py", OPTIMISEUR),
        ("11d_optimize_entry_threshold.py", OPTIMISEUR),
        ("optimize_options_multiples.py", OPTIMISEUR_NU),
        ("09_backtest.py", LISTE_STRATEGIES),
        ("10_backtest_options.py", LISTE_STRATEGIES),
        ("etape_ok.py", ETAPE_OK),
        ("etape_ko.py", ETAPE_KO),
        ("etape_lente.py", ETAPE_LENTE),
    ):
        (racine / nom).write_text(code, encoding="utf-8")
    return racine


@pytest.fixture
def banc(depot: Path, tmp_path: Path):
    annonces: list[str] = []
    contrat.set_announcer(annonces.append)
    config = Config({
        "atelier": {"racines": [str(tmp_path)], "corbeille": str(tmp_path / "corbeille")},
        "plugins": {"calculrisque": {
            "chemin": str(depot),
            "delai_s": 30.0,
            "enchainements": {
                "quotidienne": ["etape_ok.py"],
                "fragile": ["etape_ok.py", "etape_ko.py", "etape_ok.py"],
                "longue": ["etape_lente.py"],
            },
            "optimisations": {
                "stops": "11_optimize_options_stops.py",
                "seuil_entree": "11d_optimize_entry_threshold.py",
                "multiples": "optimize_options_multiples.py",
            },
        }},
    })
    contrat.set_atelier(atelier_depuis_config(config))
    registry = PluginRegistry([PLUGINS], config=config, data_root=tmp_path / "data")
    registry.load_all()
    yield {"registre": registry, "depot": depot, "annonces": annonces}
    for nom in list(registry.plugins):
        registry.unload(nom, notify=False)
    contrat.set_atelier(None)
    contrat.set_announcer(None)


def _attendre(annonces: list[str], delai: float = 20.0) -> str:
    limite = time.monotonic() + delai
    while time.monotonic() < limite:
        if annonces:
            return annonces[-1]
        time.sleep(0.02)
    return ""


# --- la frontière de l'atelier ----------------------------------------------

def test_sans_atelier_le_plugin_est_inerte(tmp_path: Path) -> None:
    # Il lance des programmes et écrit des fichiers : tant que le dossier
    # n'est pas ouvert, il refuse — comme les autres compétences sensibles.
    contrat.set_atelier(None)
    registry = PluginRegistry([PLUGINS], data_root=tmp_path / "d")
    registry.load_all()
    try:
        resultat = registry.call("mettre_a_jour", {"quoi": "quotidienne"})
        assert not resultat.ok
        assert "dossier de travail" in resultat.speak or "déclaré" in resultat.speak
    finally:
        for nom in list(registry.plugins):
            registry.unload(nom, notify=False)


def test_un_depot_hors_atelier_est_refuse(tmp_path: Path) -> None:
    config = Config({
        "atelier": {"racines": [str(tmp_path / "permis")]},
        "plugins": {"calculrisque": {"chemin": str(tmp_path / "ailleurs")}},
    })
    (tmp_path / "permis").mkdir()
    (tmp_path / "ailleurs").mkdir()
    contrat.set_atelier(atelier_depuis_config(config))
    registry = PluginRegistry([PLUGINS], config=config, data_root=tmp_path / "d")
    registry.load_all()
    try:
        assert not registry.call("mes_strategies").ok
    finally:
        for nom in list(registry.plugins):
            registry.unload(nom, notify=False)
        contrat.set_atelier(None)


# --- les enchaînements -------------------------------------------------------

def test_une_mise_a_jour_s_enchaine_et_s_annonce(banc) -> None:
    resultat = banc["registre"].call("mettre_a_jour", {"quoi": "quotidienne"})
    assert resultat.ok
    assert "terminé" in _attendre(banc["annonces"])


def test_une_etape_qui_rate_arrete_la_suite(banc) -> None:
    banc["registre"].call("mettre_a_jour", {"quoi": "fragile"})
    annonce = _attendre(banc["annonces"])
    # La troisième étape ne doit pas tourner : la deuxième a échoué.
    assert "étape 2 sur 3" in annonce and "code 3" in annonce
    etat = banc["registre"].call("etat_du_travail")
    assert "je rate" in etat.display


def test_un_nom_dicte_approximativement_est_retrouve(banc) -> None:
    # « la mise à jour quotidienne » dictée ne donne pas la clé de config.
    assert banc["registre"].call("mettre_a_jour", {"quoi": "la quotidienne"}).ok


def test_un_enchainement_inconnu_dit_ce_qu_il_connait(banc) -> None:
    resultat = banc["registre"].call("mettre_a_jour", {"quoi": "hebdomadaire"})
    assert not resultat.ok
    assert "quotidienne" in resultat.speak and "fragile" in resultat.speak


def test_un_seul_travail_a_la_fois(banc) -> None:
    # Une étape qui dure : deux optimisations en parallèle se disputeraient
    # les cœurs, et deux mises à jour la même base de données.
    banc["registre"].call("mettre_a_jour", {"quoi": "longue"})
    second = banc["registre"].call("mettre_a_jour", {"quoi": "quotidienne"})
    assert not second.ok
    assert "déjà en cours" in second.speak
    banc["registre"].call("arreter_le_travail", confirmed=True)


def test_un_travail_termine_libere_la_place(banc) -> None:
    banc["registre"].call("mettre_a_jour", {"quoi": "fragile"})
    assert "échoué" in _attendre(banc["annonces"])
    # Échoué n'est pas « en cours » : la place est libre.
    assert banc["registre"].call("mettre_a_jour", {"quoi": "quotidienne"}).ok


def test_arreter_juste_apres_avoir_lance(banc) -> None:
    # La course qui compte : entre « lance » et « arrête », le sous-processus
    # peut n'être pas encore né. L'arrêt doit prendre quand même.
    banc["registre"].call("mettre_a_jour", {"quoi": "longue"})
    arret = banc["registre"].call("arreter_le_travail", confirmed=True)
    assert arret.ok and "arrêté" in arret.speak
    # Et la place est libre tout de suite, sans attendre que le système
    # récolte le processus.
    assert banc["registre"].call("mettre_a_jour", {"quoi": "quotidienne"}).ok


def test_arreter_sans_rien_en_cours(banc) -> None:
    resultat = banc["registre"].call("arreter_le_travail", confirmed=True)
    assert "rien à arrêter" in resultat.speak


def test_arreter_demande_confirmation(banc) -> None:
    banc["registre"].call("mettre_a_jour", {"quoi": "longue"})
    assert banc["registre"].call("arreter_le_travail").needs_confirmation
    banc["registre"].call("arreter_le_travail", confirmed=True)


# --- les optimisations -------------------------------------------------------

def test_l_optimiseur_ne_recoit_que_les_drapeaux_qu_il_connait(banc) -> None:
    resultat = banc["registre"].call("optimiser", {
        "quoi": "stops", "strategie": "valuation_gap_options",
        "debut": "2020-01-01", "fin": "2024-12-31",
    })
    assert resultat.ok
    commande = resultat.display.splitlines()[0]
    assert "--strategy valuation_gap_options" in commande
    assert "--start-date 2020-01-01" in commande and "--end-date 2024-12-31" in commande
    assert "--output-csv" in commande


def test_un_optimiseur_sans_ces_drapeaux_ne_les_recoit_pas(banc) -> None:
    # Passer un drapeau inconnu ferait échouer argparse avant tout calcul.
    resultat = banc["registre"].call("optimiser", {
        "quoi": "multiples", "strategie": "valuation_gap_options", "debut": "2020-01-01",
    })
    assert resultat.ok
    commande = resultat.display.splitlines()[0]
    assert "--strategy" not in commande and "--output-csv" not in commande


def test_le_seuil_d_entree_dicte_avec_son_elision(banc) -> None:
    # « seuil d'entrée » doit tomber sur la clé seuil_entree : le « d » de
    # l'élision ne doit pas faire manquer la correspondance.
    resultat = banc["registre"].call("optimiser", {"quoi": "seuil d'entrée"})
    assert resultat.ok
    assert "11d_optimize_entry_threshold.py" in resultat.display


def test_une_date_mal_dictee_est_refusee(banc) -> None:
    resultat = banc["registre"].call("optimiser", {"quoi": "stops", "debut": "l'an dernier"})
    assert not resultat.ok and "format" in resultat.speak


def test_une_optimisation_inconnue_dit_ce_qu_elle_sait_faire(banc) -> None:
    resultat = banc["registre"].call("optimiser", {"quoi": "la couleur"})
    assert not resultat.ok and "stops" in resultat.speak


def test_lire_le_resultat_d_une_optimisation(banc) -> None:
    dossier = banc["depot"] / "data" / "optimisations"
    dossier.mkdir(parents=True)
    (dossier / "stops_20260101-120000.csv").write_text(
        "stop_loss_pct,sharpe\n35,1.42\n30,1.31\n", encoding="utf-8")
    resultat = banc["registre"].call("resultat_optimisation", {"quoi": "stops"})
    assert resultat.ok
    # La première ligne du CSV est la meilleure : les optimiseurs trient déjà.
    assert "stop_loss_pct 35" in resultat.speak
    assert "2 combinaisons" in resultat.display


def test_un_resultat_absent_se_dit(banc) -> None:
    assert not banc["registre"].call("resultat_optimisation", {"quoi": "stops"}).ok


# --- les stratégies ----------------------------------------------------------

def _creer(banc, nom: str, **arguments):
    return banc["registre"].call(
        "creer_une_strategie", {"nom": nom, **arguments}, confirmed=True
    )


def test_creer_une_strategie_ecrit_du_python_valide(banc) -> None:
    resultat = _creer(banc, "écart profond", seuil_entree=40, positions_max=15)
    assert resultat.ok, resultat.speak
    fichier = banc["depot"] / "backtest" / "strategies" / "ecart_profond.py"
    assert fichier.exists()
    code = fichier.read_text(encoding="utf-8")
    compile(code, str(fichier), "exec")
    assert 'register_strategy("ecart_profond")' in code
    assert "entry_threshold_pct: float = 40.0" in code
    assert "max_positions: int = 15" in code


def test_la_strategie_est_enregistree_dans_le_registre(banc) -> None:
    _creer(banc, "écart profond", seuil_entree=40)
    init = (banc["depot"] / "backtest" / "strategies" / "__init__.py").read_text(encoding="utf-8")
    assert "from backtest.strategies.ecart_profond import EcartProfondStrategy" in init
    assert '"EcartProfondStrategy",' in init
    # Et le registre la voit vraiment : c'est ce que la compétence vérifie.
    assert "ecart_profond" in banc["registre"].call("mes_strategies").display


def test_l_enregistrement_ne_duplique_pas_la_ligne(banc) -> None:
    _creer(banc, "une", seuil_entree=30)
    _creer(banc, "deux", seuil_entree=35)
    init = (banc["depot"] / "backtest" / "strategies" / "__init__.py").read_text(encoding="utf-8")
    assert init.count("from backtest.strategies.une import") == 1
    assert init.count("from backtest.strategies.deux import") == 1


def test_elle_n_ecrase_jamais_une_strategie_existante(banc) -> None:
    assert _creer(banc, "écart profond").ok
    second = _creer(banc, "écart profond")
    assert not second.ok and "existe déjà" in second.speak


def test_la_creation_demande_confirmation(banc) -> None:
    demande = banc["registre"].call("creer_une_strategie", {"nom": "essai"})
    assert demande.needs_confirmation
    assert not (banc["depot"] / "backtest" / "strategies" / "essai.py").exists()


@pytest.mark.parametrize(
    "arguments,attendu",
    [
        ({"conviction": "magie"}, "je connais"),
        ({"ponderation": "au pif"}, "je connais"),
    ],
)
def test_un_reglage_inconnu_est_refuse(banc, arguments, attendu) -> None:
    resultat = _creer(banc, "essai", **arguments)
    assert not resultat.ok and attendu in resultat.speak


def test_un_nom_sans_lettre_est_refuse(banc) -> None:
    assert not _creer(banc, "??? !!!").ok


def test_le_filtre_de_secteurs_entre_dans_le_code(banc) -> None:
    resultat = _creer(banc, "techno", secteurs="Information Technology, Health Care")
    assert resultat.ok
    code = (banc["depot"] / "backtest" / "strategies" / "techno.py").read_text(encoding="utf-8")
    assert "'Information Technology', 'Health Care'" in code
    compile(code, "techno.py", "exec")


def test_l_equiponderation_change_le_code_produit(banc) -> None:
    _creer(banc, "egale", ponderation="egale")
    code = (banc["depot"] / "backtest" / "strategies" / "egale.py").read_text(encoding="utf-8")
    assert "pd.Series(1.0 / len(candidates)" in code
    assert "capped_weights(conviction)" not in code


def test_l_excedent_sectoriel_change_la_conviction(banc) -> None:
    _creer(banc, "neutre", conviction="excedent_sectoriel")
    code = (banc["depot"] / "backtest" / "strategies" / "neutre.py").read_text(encoding="utf-8")
    assert 'groupby("sector")["gap_pct"].transform("median")' in code


def test_un_init_de_forme_inattendue_n_est_pas_touche_a_l_aveugle(banc) -> None:
    init = banc["depot"] / "backtest" / "strategies" / "__init__.py"
    init.write_text("# un fichier qui n'a pas la forme attendue\n", encoding="utf-8")
    resultat = _creer(banc, "essai")
    assert not resultat.ok
    assert "forme attendue" in resultat.speak
    assert init.read_text(encoding="utf-8").startswith("# un fichier")


# --- les backtests -----------------------------------------------------------

def test_backtester_construit_la_bonne_commande(banc) -> None:
    assert banc["registre"].call("backtester", {"strategie": "valuation_gap_dcf"}).ok
    banc["registre"].call("arreter_le_travail", confirmed=True)
    resultat = banc["registre"].call(
        "backtester", {"strategie": "valuation gap options", "options": True}
    )
    assert resultat.ok and "options" in resultat.speak


def test_backtester_sans_strategie_refuse(banc) -> None:
    assert not banc["registre"].call("backtester", {"strategie": "  "}).ok

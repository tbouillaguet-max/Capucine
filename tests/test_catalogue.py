"""Le catalogue d'API : lire les signatures, et ne montrer que ce qui sert."""

from __future__ import annotations

from pathlib import Path

import pytest

from capucine.core.catalogue import Catalogue, _premiere_phrase, _signature, depuis_config
from capucine.core.config import Config

MODULE = '''"""Un module d'exemple."""

import math


def capped_weights(conviction, cap_pct: float | None = None, max_iter: int = 20) -> "pd.Series":
    """Poids proportionnels à `conviction`, aucun ne dépassant cap_pct %.

    Détail qui ne doit PAS entrer dans le résumé.
    """
    return conviction


def _interne(x):
    """Ne doit pas apparaître : elle est privée."""
    return x


async def realized_volatility(close_history, lookback_days: int = 20) -> float | None:
    """Volatilité annualisée des rendements log quotidiens."""
    return None


class Portefeuille:
    """Un portefeuille de positions."""

    def __init__(self, capital: float = 0.0) -> None:
        """Ouvre un portefeuille."""

    def valoriser(self, cours) -> float:
        """Valorise les positions au cours du jour."""
        return 0.0

    def _prive(self) -> None:
        """Invisible."""
'''

TEST_QUI_POLLUE = '''
def test_le_plafond_par_position_reste_applique():
    """Vérifie le plafond par position."""
'''


@pytest.fixture
def depot(tmp_path: Path) -> Path:
    racine = tmp_path / "projet"
    (racine / "sous").mkdir(parents=True)
    (racine / "outils.py").write_text(MODULE, encoding="utf-8")
    (racine / "sous" / "autre.py").write_text(MODULE, encoding="utf-8")
    (racine / "tests").mkdir()
    (racine / "tests" / "test_poids.py").write_text(TEST_QUI_POLLUE, encoding="utf-8")
    (racine / "casse.py").write_text("def (((\n", encoding="utf-8")
    (racine / "__pycache__").mkdir()
    (racine / "__pycache__" / "vieux.py").write_text(MODULE, encoding="utf-8")
    return racine


# --- la lecture --------------------------------------------------------------

def test_il_lit_les_signatures_sans_importer(depot: Path) -> None:
    # Importer exécuterait le code du dépôt et exigerait ses dépendances.
    catalogue = Catalogue(depot)
    assert catalogue.construire() > 0
    entree = catalogue.par_nom("capped_weights")
    assert entree is not None
    assert entree.signature == (
        "(conviction, cap_pct: float | None = None, max_iter: int = 20) -> 'pd.Series'"
    )
    assert entree.genre == "fonction"


def test_le_resume_s_arrete_a_la_premiere_phrase(depot: Path) -> None:
    catalogue = Catalogue(depot)
    entree = catalogue.par_nom("capped_weights")
    assert entree.resume.startswith("Poids proportionnels")
    assert "ne doit PAS" not in entree.resume


def test_les_privees_sont_ecartees_par_defaut(depot: Path) -> None:
    catalogue = Catalogue(depot)
    catalogue.construire()
    assert catalogue.par_nom("_interne") is None
    # `__init__` reste : sans elle on ne sait pas construire la classe.
    assert catalogue.par_nom("Portefeuille.__init__") is not None


def test_les_privees_peuvent_etre_demandees(depot: Path) -> None:
    catalogue = Catalogue(depot, privees=True)
    catalogue.construire()
    assert catalogue.par_nom("_interne") is not None


def test_les_tests_ne_sont_pas_de_la_surface_d_api(depot: Path) -> None:
    # Personne n'appelle un test depuis du code neuf, mais son nom capte
    # toutes les recherches sur « plafond par position ».
    catalogue = Catalogue(depot)
    catalogue.construire()
    assert catalogue.par_nom("test_le_plafond_par_position_reste_applique") is None
    assert Catalogue(depot, tests=True).par_nom(
        "test_le_plafond_par_position_reste_applique"
    ) is not None


def test_un_fichier_illisible_ne_prive_pas_du_reste(depot: Path) -> None:
    catalogue = Catalogue(depot)
    assert catalogue.construire() > 0
    assert catalogue.par_nom("capped_weights") is not None


def test_les_dossiers_de_cache_sont_ignores(depot: Path) -> None:
    catalogue = Catalogue(depot)
    catalogue.construire()
    assert not any("__pycache__" in entree.fichier for entree in catalogue._entrees)


def test_les_methodes_portent_le_nom_de_leur_classe(depot: Path) -> None:
    catalogue = Catalogue(depot)
    catalogue.construire()
    entree = catalogue.par_nom("Portefeuille.valoriser")
    assert entree is not None and entree.genre == "methode"
    assert entree.qualifie == "outils.Portefeuille.valoriser"


def test_une_relecture_sans_changement_ne_coute_rien(depot: Path) -> None:
    catalogue = Catalogue(depot)
    premier = catalogue.construire()
    empreintes = dict(catalogue._empreintes)
    assert catalogue.construire() == premier
    assert catalogue._empreintes == empreintes


def test_un_fichier_modifie_est_relu(depot: Path) -> None:
    catalogue = Catalogue(depot)
    catalogue.construire()
    assert catalogue.par_nom("toute_neuve") is None
    (depot / "outils.py").write_text(
        MODULE + '\n\ndef toute_neuve(x: int) -> int:\n    """Neuve."""\n    return x\n',
        encoding="utf-8",
    )
    import os
    os.utime(depot / "outils.py", (0, 0))      # date différente, garantie
    catalogue.construire()
    assert catalogue.par_nom("toute_neuve") is not None


# --- le rendu des signatures -------------------------------------------------

@pytest.mark.parametrize(
    "source,attendu",
    [
        ("def f(a, /, b, *c, d=1, **e) -> int: ...", "(a, /, b, *c, d = 1, **e) -> int"),
        ("def f(): ...", "()"),
        ("def f(self, x: int) -> None: ...", "(x: int) -> None"),
        ("async def f(*, y: str = 'a'): ...", "(*, y: str = 'a')"),
    ],
)
def test_la_signature_est_rendue_lisible(source, attendu) -> None:
    import ast

    assert _signature(ast.parse(source).body[0]) == attendu


def test_un_resume_qui_commence_par_un_parametre(  ) -> None:
    # « signals : signaux connus… » ne doit pas se réduire à « signals ».
    assert _premiere_phrase("signals : les signaux connus à la date courante") != "signals"


# --- la sélection : ne montrer que ce qui sert -------------------------------

def test_une_demande_sans_rapport_ne_rend_rien(depot: Path) -> None:
    # Injecter douze fonctions au hasard dans le contexte d'un 7B est PIRE
    # que de n'en injecter aucune.
    catalogue = Catalogue(depot)
    assert catalogue.chercher("envoie un mail à mon frère") == []
    assert catalogue.chercher("quelle heure est-il") == []


def test_nommer_une_fonction_la_sort_en_tete(depot: Path) -> None:
    catalogue = Catalogue(depot)
    trouvees = catalogue.chercher("utilise capped_weights")
    assert trouvees and trouvees[0].nom == "capped_weights"


def test_le_tiret_bas_ne_bloque_pas_la_correspondance(depot: Path) -> None:
    # `normalize` garde le tiret bas — c'est un caractère de mot pour \w.
    catalogue = Catalogue(depot)
    assert any(e.nom == "capped_weights" for e in catalogue.chercher("capped_weights"))


def test_un_cognat_francais_trouve_l_identifiant_anglais(depot: Path) -> None:
    # Les identifiants sont en anglais, les demandes en français : sans
    # l'appariement par cognat, « volatilité » ne trouve pas « volatility ».
    catalogue = Catalogue(depot)
    trouvees = catalogue.chercher("volatilité annualisée des rendements")
    assert any(entree.nom == "realized_volatility" for entree in trouvees)


def test_la_reference_est_bornee_en_taille(depot: Path) -> None:
    catalogue = Catalogue(depot, caracteres_max=120)
    reference = catalogue.reference("poids proportionnels plafonnés")
    # Une entrée, pas douze — et jamais vide : la meilleure passe toujours,
    # réduite à sa déclaration si besoin.
    assert reference and "capped_weights" in reference
    assert reference.count("from ") == 1      # une entrée, pas douze


def test_la_reference_est_vide_quand_rien_ne_correspond(depot: Path) -> None:
    assert Catalogue(depot).reference("envoie un mail") == ""


def test_la_reference_montre_la_signature_et_le_role(depot: Path) -> None:
    reference = Catalogue(depot).reference("poids proportionnels plafonnés")
    # Du Python VALIDE : une ligne d'import, puis la forme de l'appel.
    # « def outils.capped_weights(...) » ne se déclare ni ne s'appelle.
    assert "from outils import capped_weights" in reference
    assert "capped_weights(conviction, cap_pct" in reference
    assert "def " not in reference
    assert "→" in reference


# --- statistiques et configuration -------------------------------------------

def test_les_statistiques_comptent_par_genre(depot: Path) -> None:
    stats = Catalogue(depot).statistiques()
    assert stats["total"] > 0
    assert stats["fonction"] >= 2 and stats["classe"] >= 1 and stats["methode"] >= 1


def test_le_catalogue_se_desactive(tmp_path: Path) -> None:
    config = Config({"catalogue": {"active": False, "racine": str(tmp_path)}})
    assert depuis_config(config) is None


def test_sans_racine_il_n_y_a_pas_de_catalogue() -> None:
    assert depuis_config(Config({"catalogue": {}})) is None


# --- la mesure longue, et sa limite ------------------------------------------

DEPOT_REEL = '''
def inflation_adjusted_gap(gap_pct, published_date, horizon_years: float):
    """Écart de valorisation corrigé de l'inflation attendue sur l'horizon."""


def capped_weights(conviction, cap_pct: float | None = None):
    """Poids proportionnels à `conviction`, aucun ne dépassant cap_pct % du portefeuille."""


def bs_price(spot: float, strike: float, t_years: float, vol: float, option_type: str):
    """Prix Black-Scholes-Merton."""


def charger_les_cours(chemin):
    """Charge les cours quotidiens depuis un fichier."""
'''


@pytest.fixture
def realiste(tmp_path: Path) -> Catalogue:
    racine = tmp_path / "depot"
    racine.mkdir()
    (racine / "outils.py").write_text(DEPOT_REEL, encoding="utf-8")
    return Catalogue(racine)


def test_une_description_longue_trouve_quand_meme(realiste: Catalogue) -> None:
    """Le défaut qui rendait le catalogue inerte sur les vraies demandes.

    Le score symétrique divisait par le nombre de mots de la demande : une
    description longue et naturelle — c'est-à-dire ce qu'on écrit vraiment —
    était pénalisée, et le catalogue ne rendait rien.
    """
    longue = (
        "Écris une fonction corriger(signaux) qui corrige la colonne gap_pct "
        "d'un DataFrame de l'inflation attendue, en utilisant la fonction du "
        "projet prévue pour ça."
    )
    trouvees = realiste.chercher(longue)
    assert any(entree.nom == "inflation_adjusted_gap" for entree in trouvees)


def test_une_description_longue_de_pricing_trouve(realiste: Catalogue) -> None:
    longue = (
        "Écris une fonction prime(spot, strike, annees, vol) qui rend le prix "
        "d'un call, en utilisant la fonction de valorisation Black-Scholes du "
        "projet."
    )
    trouvees = realiste.chercher(longue)
    assert trouvees and trouvees[0].nom == "bs_price"


def test_une_description_longue_hors_sujet_ne_trouve_rien(realiste: Catalogue) -> None:
    longue = (
        "Écris une fonction mediane(valeurs) qui rend la médiane d'une liste "
        "de nombres, sans utiliser statistics."
    )
    assert realiste.chercher(longue) == []


@pytest.mark.xfail(
    reason="limite lexicale assumée : « plafonnant » et « dépassant » disent la "
           "même chose sans partager un mot. C'est un rapprochement sémantique, "
           "hors de portée d'un score lexical — c'est là que l'index de "
           "semantique.py gagnerait sa place.",
    strict=True,
)
def test_limite_le_synonyme_sans_mot_commun(realiste: Catalogue) -> None:
    longue = (
        "Écris une fonction repartir(convictions) qui pondère un dictionnaire "
        "en plafonnant chaque poids, en réutilisant la fonction du projet."
    )
    trouvees = realiste.chercher(longue)
    assert any(entree.nom == "capped_weights" for entree in trouvees)


def test_une_entree_se_rend_en_python_valide(depot: Path) -> None:
    """Le défaut qui rendait la référence inutilisable par un modèle de code.

    « def outils.capped_weights(...) » n'est pas du Python : on ne déclare
    pas `def a.b.c()`, on ne l'appelle pas non plus. Le modèle recevait une
    forme impossible, sans savoir quoi importer.
    """
    entree = Catalogue(depot).par_nom("capped_weights")
    assert entree.importation() == "from outils import capped_weights"
    assert entree.appel().startswith("capped_weights(conviction")
    # L'import est du Python compilable, tel quel.
    compile(entree.importation(), "<ref>", "exec")


def test_une_methode_montre_l_import_de_sa_classe(depot: Path) -> None:
    entree = Catalogue(depot).par_nom("Portefeuille.valoriser")
    assert entree.importation() == "from outils import Portefeuille"
    assert entree.appel().startswith("Portefeuille.valoriser(cours)")


def test_un_module_non_importable_ne_ment_pas(tmp_path: Path) -> None:
    # `from 08_recuperation import x` est une erreur de syntaxe : montrer cet
    # import apprendrait au modèle à en écrire un, pire que de n'en montrer
    # aucun.
    racine = tmp_path / "p"
    racine.mkdir()
    (racine / "08_recuperation.py").write_text(
        'def chercher(x):\n    """Cherche."""\n', encoding="utf-8")
    entree = Catalogue(racine).par_nom("chercher")
    assert entree.importation().startswith("# défini dans")
    assert "from" not in entree.importation()


# --- le code des DÉPENDANCES n'est pas votre API ------------------------------

def test_un_environnement_virtuel_est_ecarte_quel_que_soit_son_nom(tmp_path: Path) -> None:
    """Le défaut qui a fait écrire au modèle un import de bibliothèque tierce.

    Sur une vraie machine, le dépôt hébergeait son environnement virtuel sous
    un nom quelconque. Le catalogue y a trouvé `narwhals`, l'a montré au
    modèle, et celui-ci a écrit
    ``from narwhals._compliant.any_namespace import DateTimeNamespace``
    pour formater une date — cassant une tâche qui passait sans catalogue.
    Injecter des fonctions hors sujet est bien pire que n'en injecter aucune.
    """
    racine = tmp_path / "projet"
    (racine / "env311" / "Lib" / "site-packages" / "narwhals").mkdir(parents=True)
    (racine / "outils.py").write_text(
        'def ma_fonction(x):\n    """La vraie."""\n', encoding="utf-8")
    # `pyvenv.cfg` est écrit par venv et virtualenv quel que soit le nom du
    # dossier : c'est le seul test qui attrape un « env311 ».
    (racine / "env311" / "pyvenv.cfg").write_text("home = /usr", encoding="utf-8")
    (racine / "env311" / "Lib" / "site-packages" / "narwhals" / "espace.py").write_text(
        'class DateTimeNamespace:\n    """Tierce."""\n', encoding="utf-8")

    catalogue = Catalogue(racine)
    catalogue.construire()
    assert catalogue.par_nom("ma_fonction") is not None
    assert catalogue.par_nom("DateTimeNamespace") is None


def test_un_site_packages_sans_venv_est_ecarte_aussi(tmp_path: Path) -> None:
    racine = tmp_path / "projet"
    (racine / "vendor" / "site-packages" / "truc").mkdir(parents=True)
    (racine / "app.py").write_text('def mienne():\n    """Mienne."""\n', encoding="utf-8")
    (racine / "vendor" / "site-packages" / "truc" / "m.py").write_text(
        'def fonction_tierce():\n    """Tierce."""\n', encoding="utf-8")

    catalogue = Catalogue(racine)
    catalogue.construire()
    assert catalogue.par_nom("mienne") is not None
    assert catalogue.par_nom("fonction_tierce") is None


def test_un_dossier_lib_legitime_n_est_pas_ecarte(tmp_path: Path) -> None:
    # « lib » et « Lib » ne sont pas dans la liste noire : beaucoup de projets
    # rangent leur propre code dans lib/, et l'écarter par son nom serait pire
    # que le mal. C'est la STRUCTURE du venv qui tranche, pas le nom.
    racine = tmp_path / "projet"
    (racine / "lib").mkdir(parents=True)
    (racine / "lib" / "interne.py").write_text(
        'def utilitaire_maison():\n    """À nous."""\n', encoding="utf-8")

    assert Catalogue(racine).par_nom("utilitaire_maison") is not None


def test_l_elagage_ne_descend_pas_dans_l_environnement(tmp_path: Path) -> None:
    # L'élagage se fait à la descente : les milliers de fichiers d'un venv ne
    # doivent pas être seulement filtrés, ils ne doivent pas être VISITÉS.
    racine = tmp_path / "projet"
    profond = racine / ".venv" / "Lib" / "site-packages"
    profond.mkdir(parents=True)
    for numero in range(50):
        (profond / f"m{numero}.py").write_text(
            f'def tierce_{numero}():\n    """Tierce."""\n', encoding="utf-8")
    (racine / "app.py").write_text('def mienne():\n    """Mienne."""\n', encoding="utf-8")

    catalogue = Catalogue(racine)
    catalogue.construire()
    assert catalogue.statistiques()["fichiers"] == 1

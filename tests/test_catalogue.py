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
    assert reference.count("def ") + reference.count("class ") == 1


def test_la_reference_est_vide_quand_rien_ne_correspond(depot: Path) -> None:
    assert Catalogue(depot).reference("envoie un mail") == ""


def test_la_reference_montre_la_signature_et_le_role(depot: Path) -> None:
    reference = Catalogue(depot).reference("poids proportionnels plafonnés")
    assert "def " in reference and "cap_pct" in reference
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

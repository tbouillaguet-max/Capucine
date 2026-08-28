"""Isolation en sous-processus : la seule façon de tuer vraiment un plugin.

Ces tests lancent de vrais interpréteurs ; ils sont donc plus lents que les
autres. C'est le prix de la garantie qu'ils vérifient.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from capucine.core.config import Config
from capucine.core.registry import PluginRegistry

PLUGIN_ISOLE = '''
import time
from capucine.plugin import data_dir, get_config, skill

CONFIG_DEFAULTS = {"facteur": 2}

@skill(description="Multiplie, dans un sous-processus.", isolate=True, timeout=20.0)
def multiplier(n: int = 1) -> str:
    print("du bruit sur la sortie standard, qui ne doit rien corrompre")
    return f"{n * int(get_config('facteur', 2))}"

@skill(description="Ne finit jamais.", isolate=True, timeout=0.5)
def bloquer() -> str:
    while True:
        time.sleep(0.05)

@skill(description="Explose dans l'enfant.", isolate=True, timeout=20.0)
def exploser() -> str:
    raise RuntimeError("boum isolé")

@skill(description="Rend un objet impossible à sérialiser.", isolate=True, timeout=20.0)
def indeserialisable():
    return lambda x: x

@skill(description="Écrit dans son dossier de données.", isolate=True, timeout=20.0)
def ecrire() -> str:
    chemin = data_dir() / "isole.txt"
    chemin.write_text("écrit depuis le sous-processus", encoding="utf-8")
    return str(chemin)
'''


@pytest.fixture
def registre_isole(ecrire_plugin, dossier_plugins, tmp_path: Path):
    ecrire_plugin("isole.py", PLUGIN_ISOLE)
    registry = PluginRegistry(
        [dossier_plugins],
        config=Config({"plugins": {"isole": {"facteur": 3}}}),
        data_root=tmp_path / "data",
        isolate_startup_s=8.0,
    )
    registry.load_all()
    yield registry
    for nom in list(registry.plugins):
        registry.unload(nom, notify=False)


def test_une_competence_isolee_rend_bien_son_resultat(registre_isole) -> None:
    resultat = registre_isole.call("multiplier", {"n": 14})
    assert resultat.ok
    # La configuration du plugin traverse la frontière du processus.
    assert resultat.speak == "42"


def test_le_bavardage_du_plugin_ne_corrompt_pas_la_reponse(registre_isole) -> None:
    # Le résultat transite par un fichier, pas par la sortie standard :
    # un print() dans le plugin est sans effet.
    assert registre_isole.call("multiplier", {"n": 1}).speak == "3"


def test_une_boucle_infinie_est_reellement_tuee(registre_isole) -> None:
    # C'est la seule chose que le mode thread ne sait pas faire.
    depart = time.perf_counter()
    resultat = registre_isole.call("bloquer")
    duree = time.perf_counter() - depart

    assert not resultat.ok
    assert "trop de temps" in resultat.speak
    # Le délai du plugin, plus la marge d'amorçage : pas indéfiniment.
    assert duree < 20.0


def test_une_exception_dans_l_enfant_remonte_proprement(registre_isole) -> None:
    resultat = registre_isole.call("exploser")
    assert not resultat.ok
    assert resultat.speak == "Je n'ai pas pu exécuter cette commande."


def test_un_retour_non_serialisable_donne_un_message_clair(registre_isole) -> None:
    resultat = registre_isole.call("indeserialisable")
    assert not resultat.ok


def test_le_dossier_de_donnees_est_disponible_dans_l_enfant(registre_isole) -> None:
    resultat = registre_isole.call("ecrire")
    assert resultat.ok
    chemin = Path(resultat.speak)
    assert chemin.parent.name == "isole"
    assert chemin.read_text(encoding="utf-8").startswith("écrit")


def test_des_arguments_non_serialisables_sont_refuses_avant_le_lancement() -> None:
    from capucine.core.errors import SkillCrashed
    from capucine.core.isolation import run_isolated

    with pytest.raises(SkillCrashed, match="sérialisables"):
        # Une lambda ne traverse pas la frontière du processus ; on le dit
        # avant de payer le lancement d'un interpréteur.
        run_isolated(
            Path("inexistant.py"), "m", "f", {"rappel": lambda x: x},
            timeout=1.0, plugin="essai",
        )


def test_le_mode_thread_reste_le_defaut(ecrire_plugin, dossier_plugins, tmp_path) -> None:
    # L'isolation coûte un processus par appel : elle se demande explicitement.
    ecrire_plugin("ordinaire.py", '''
from capucine.plugin import skill

@skill(description="Ordinaire.")
def ordinaire() -> str:
    return "ok"
''')
    registry = PluginRegistry([dossier_plugins], data_root=tmp_path / "data")
    registry.load_all()
    assert registry.get("ordinaire").isolate is False

"""Registre de plugins : découverte, isolation des pannes, cycle de vie.

C'est le cœur du projet : un fichier déposé dans le dossier suffit, et un
plugin fautif ne doit jamais faire tomber Lily.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lily.core.config import Config
from lily.core.registry import PluginRegistry

PLUGIN_SIMPLE = '''
from lily.plugin import skill

@skill(description="Dit bonjour.", examples=["dis bonjour"])
def saluer(nom: str = "toi") -> str:
    """Salue quelqu'un."""
    return f"Bonjour {nom}."
'''


def test_un_fichier_depose_suffit(ecrire_plugin, registre) -> None:
    ecrire_plugin("salutations.py", PLUGIN_SIMPLE)
    registry = registre()
    registry.load_all()

    assert "saluer" in registry.skills
    spec = registry.get("saluer")
    assert spec.plugin == "salutations"
    # Le schéma d'outil est généré sans que le plugin ait écrit une ligne de JSON.
    assert spec.tool_schema["function"]["parameters"]["properties"]["nom"]["type"] == "string"
    assert "dis bonjour" in spec.tool_schema["function"]["description"]

    resultat = registry.call("saluer", {"nom": "Lily"})
    assert resultat.ok
    assert resultat.speak == "Bonjour Lily."


def test_un_retour_dict_dissocie_la_parole_et_le_journal(ecrire_plugin, registre) -> None:
    ecrire_plugin("dual.py", '''
from lily.plugin import skill

@skill(description="Deux sorties.")
def deux() -> dict:
    return {"speak": "Il fait douze degrés.", "display": "temp=12.0C src=capteur"}
''')
    registry = registre()
    registry.load_all()
    resultat = registry.call("deux")
    assert resultat.speak == "Il fait douze degrés."
    assert resultat.display == "temp=12.0C src=capteur"


def test_un_plugin_casse_a_l_import_est_ignore_les_autres_survivent(ecrire_plugin, registre) -> None:
    ecrire_plugin("salutations.py", PLUGIN_SIMPLE)
    ecrire_plugin("casse.py", "raise ValueError('je casse à l\\'import')\n")

    registry = registre()
    registry.load_all()

    assert "saluer" in registry.skills          # le bon plugin fonctionne
    echecs = {r.name: r for r in registry.failures()}
    assert "casse" in echecs
    assert "je casse" in echecs["casse"].error
    # Le module fautif ne reste pas dans sys.modules.
    assert "lily.plugins.casse" not in sys.modules


def test_une_erreur_de_syntaxe_est_traitee_comme_le_reste(ecrire_plugin, registre) -> None:
    ecrire_plugin("bancal.py", "def f(:\n")
    registry = registre()
    registry.load_all()
    assert [r.name for r in registry.failures()] == ["bancal"]


def test_une_dependance_manquante_nomme_le_paquet(ecrire_plugin, registre) -> None:
    ecrire_plugin("exotique.py", "import paquet_qui_n_existe_pas\n")
    registry = registre()
    registry.load_all()

    (echec,) = registry.failures()
    # Jamais d'installation automatique : on nomme le paquet et la commande.
    assert echec.missing_package == "paquet_qui_n_existe_pas"
    assert "pip install paquet_qui_n_existe_pas" in echec.error


def test_un_plugin_lent_est_abandonne_et_lily_repond(ecrire_plugin, registre) -> None:
    ecrire_plugin("lent.py", '''
import time
from lily.plugin import skill

@skill(description="Ne finit jamais.", timeout=0.2)
def trainer() -> str:
    time.sleep(30)
    return "jamais"
''')
    registry = registre()
    registry.load_all()

    resultat = registry.call("trainer")
    assert not resultat.ok
    assert "trop de temps" in resultat.speak
    # Le registre reste utilisable après coup.
    assert "trainer" in registry.skills


def test_un_plugin_qui_leve_ne_fait_pas_tomber_lily(ecrire_plugin, registre) -> None:
    ecrire_plugin("explose.py", '''
from lily.plugin import skill

@skill(description="Explose.")
def exploser() -> str:
    raise RuntimeError("boum")
''')
    registry = registre()
    registry.load_all()

    resultat = registry.call("exploser")
    assert not resultat.ok
    assert resultat.speak == "Je n'ai pas pu exécuter cette commande."


def test_un_plugin_qui_appelle_sys_exit_ne_tue_pas_le_processus(ecrire_plugin, registre) -> None:
    # SystemExit n'hérite pas d'Exception : sans capture de BaseException à la
    # frontière, ce plugin arrêterait Lily.
    ecrire_plugin("suicidaire.py", '''
import sys
from lily.plugin import skill

@skill(description="Quitte.")
def quitter() -> str:
    sys.exit("au revoir")
''')
    registry = registre()
    registry.load_all()

    resultat = registry.call("quitter")
    assert not resultat.ok


def test_quarantaine_apres_plusieurs_echecs(ecrire_plugin, registre) -> None:
    ecrire_plugin("explose.py", '''
from lily.plugin import skill

@skill(description="Explose.")
def exploser() -> str:
    raise RuntimeError("boum")
''')
    registry = registre(quarantine_after=2)
    registry.load_all()

    registry.call("exploser")
    registry.call("exploser")
    assert registry.get("exploser").quarantined
    # Un skill en quarantaine n'est plus proposé au modèle.
    assert registry.tool_schemas() == []
    resultat = registry.call("exploser")
    assert "désactivée" in resultat.speak

    registry.reset_quarantine("exploser")
    assert not registry.get("exploser").quarantined


def test_les_hooks_de_cycle_de_vie_sont_appeles(ecrire_plugin, registre, tmp_path: Path) -> None:
    trace = tmp_path / "trace.txt"
    ecrire_plugin("cycle.py", f'''
from pathlib import Path
from lily.plugin import skill

TRACE = Path(r"{trace}")

def on_load():
    TRACE.write_text("charge\\n", encoding="utf-8")

def on_unload():
    with TRACE.open("a", encoding="utf-8") as f:
        f.write("decharge\\n")

@skill(description="Rien.")
def rien() -> str:
    return "rien"
''')
    registry = registre()
    registry.load_all()
    assert trace.read_text(encoding="utf-8") == "charge\n"

    registry.unload("cycle")
    assert trace.read_text(encoding="utf-8") == "charge\ndecharge\n"
    assert "rien" not in registry.skills


def test_un_on_load_qui_echoue_ecarte_le_plugin(ecrire_plugin, registre) -> None:
    ecrire_plugin("mauvais_demarrage.py", '''
from lily.plugin import skill

def on_load():
    raise OSError("port déjà utilisé")

@skill(description="Rien.")
def rien() -> str:
    return "rien"
''')
    registry = registre()
    registry.load_all()
    assert "rien" not in registry.skills
    (echec,) = registry.failures()
    assert "on_load" in echec.error


def test_config_defaults_surchargee_par_le_fichier(ecrire_plugin, registre) -> None:
    ecrire_plugin("reglable.py", '''
from lily.plugin import get_config, skill

CONFIG_DEFAULTS = {"unite": "celsius", "precision": 1}

AU_CHARGEMENT = get_config("unite")   # get_config marche dès l'import

@skill(description="Donne les réglages.")
def reglages() -> str:
    return f"{get_config('unite')}/{get_config('precision')}/{AU_CHARGEMENT}"
''')
    config = Config({"plugins": {"reglable": {"unite": "fahrenheit"}}})
    registry = registre(config=config)
    registry.load_all()

    # La valeur du fichier gagne, celle du module reste pour le reste.
    assert registry.call("reglages").speak == "fahrenheit/1/fahrenheit"


def test_data_dir_est_propre_a_chaque_plugin(ecrire_plugin, registre, tmp_path: Path) -> None:
    ecrire_plugin("stockeur.py", '''
from lily.plugin import data_dir, skill

@skill(description="Écrit un fichier.")
def ecrire() -> str:
    chemin = data_dir() / "note.txt"
    chemin.write_text("contenu", encoding="utf-8")
    return str(chemin)
''')
    registry = registre()
    registry.load_all()
    chemin = Path(registry.call("ecrire").speak)
    assert chemin.parent.name == "stockeur"
    assert chemin.read_text(encoding="utf-8") == "contenu"


def test_un_nom_de_fichier_accentue_fonctionne(ecrire_plugin, registre) -> None:
    # Le critère d'acceptation du projet utilise littéralement « plugins/dés.py ».
    ecrire_plugin("dés.py", '''
import random
from lily.plugin import skill

@skill(description="Lance un dé.", examples=["lance un dé"])
def lancer_de(faces: int = 6) -> str:
    """Lance un dé à N faces."""
    return f"{random.randint(1, faces)}"
''')
    registry = registre()
    registry.load_all()

    assert "lancer_de" in registry.skills
    assert registry.get("lancer_de").plugin == "dés"
    resultat = registry.call("lancer_de", {"faces": "vingt"})
    assert resultat.ok and 1 <= int(resultat.speak) <= 20


def test_un_nom_de_skill_accentue_est_expose_en_ascii(ecrire_plugin, registre) -> None:
    ecrire_plugin("accents.py", '''
from lily.plugin import skill

@skill(description="Éteint la lumière.")
def éteindre() -> str:
    return "éteint"
''')
    registry = registre()
    registry.load_all()
    # Le nom vu par le modèle est ASCII ; les accents restent dans la description.
    assert "eteindre" in registry.skills
    assert registry.call("eteindre").speak == "éteint"


def test_rechargement_prend_en_compte_les_modifications(ecrire_plugin, registre) -> None:
    chemin = ecrire_plugin("evolutif.py", '''
from lily.plugin import skill

@skill(description="Version 1.")
def version() -> str:
    return "un"
''')
    registry = registre()
    registry.load_all()
    assert registry.call("version").speak == "un"

    chemin.write_text('''
from lily.plugin import skill

@skill(description="Version 2.")
def version() -> str:
    return "deux"

@skill(description="Nouveauté.")
def nouveaute() -> str:
    return "neuf"
''', encoding="utf-8")
    registry.reload_file(chemin)

    assert registry.call("version").speak == "deux"
    assert "nouveaute" in registry.skills


def test_le_rechargement_retire_les_skills_disparus(ecrire_plugin, registre) -> None:
    chemin = ecrire_plugin("retrecit.py", '''
from lily.plugin import skill

@skill(description="A.")
def a() -> str: return "a"

@skill(description="B.")
def b() -> str: return "b"
''')
    registry = registre()
    registry.load_all()
    assert {"a", "b"} <= set(registry.skills)

    chemin.write_text('''
from lily.plugin import skill

@skill(description="A.")
def a() -> str: return "a"
''', encoding="utf-8")
    registry.reload_file(chemin)
    assert "a" in registry.skills and "b" not in registry.skills


def test_le_rappel_on_change_annonce_les_nouveautes(ecrire_plugin, registre) -> None:
    evenements: list[tuple[list[str], list[str]]] = []
    registry = registre()
    registry.on_change = lambda ajoutes, retires: evenements.append((ajoutes, retires))

    chemin = ecrire_plugin("tardif.py", PLUGIN_SIMPLE)
    registry.load_file(chemin)
    assert evenements == [(["saluer"], [])]

    registry.unload("tardif")
    assert evenements[-1] == ([], ["saluer"])


def test_un_skill_au_schema_impossible_n_emporte_pas_ses_voisins(ecrire_plugin, registre) -> None:
    ecrire_plugin("mixte.py", '''
from lily.plugin import skill

@skill(description="Impossible à décrire.")
def variadique(*args) -> str:
    return "jamais"

@skill(description="Correcte.")
def correcte() -> str:
    return "ok"
''')
    registry = registre()
    registry.load_all()
    assert "correcte" in registry.skills
    assert "variadique" not in registry.skills


def test_un_argument_obligatoire_manquant_ne_lance_pas_le_plugin(ecrire_plugin, registre) -> None:
    ecrire_plugin("exigeant.py", '''
from lily.plugin import skill

APPELS = []

@skill(description="Exige une ville.")
def meteo(ville: str) -> str:
    APPELS.append(ville)
    return ville
''')
    registry = registre()
    registry.load_all()

    resultat = registry.call("meteo", {})
    assert not resultat.ok
    assert "ville" in resultat.speak
    assert sys.modules["lily.plugins.exigeant"].APPELS == []


def test_les_fichiers_prefixes_sont_ignores(ecrire_plugin, registre) -> None:
    ecrire_plugin("_utilitaires.py", "VALEUR = 1\n")
    ecrire_plugin("salutations.py", PLUGIN_SIMPLE)
    registry = registre()
    registry.load_all()
    assert set(registry.plugins) == {"salutations"}


def test_un_dossier_absent_ne_fait_pas_echouer_le_chargement(tmp_path: Path) -> None:
    registry = PluginRegistry([tmp_path / "nulle-part"], data_root=tmp_path / "data")
    registry.load_all()
    assert registry.skills == {}


def test_un_skill_asynchrone_est_supporte(ecrire_plugin, registre) -> None:
    # Les plugins restent des fonctions synchrones ordinaires ; « async def »
    # est accepté en bonus, sans que le contrat change.
    ecrire_plugin("asynchrone.py", """
import asyncio
from lily.plugin import skill

@skill(description="Attend un peu.")
async def patienter() -> str:
    await asyncio.sleep(0.01)
    return "fini"
""")
    registry = registre()
    registry.load_all()
    assert registry.call("patienter").speak == "fini"

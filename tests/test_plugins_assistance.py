"""Les cinq compétences d'assistance : mémoire, recherche, fichiers, Python, projet.

Elles élargissent nettement ce que Capucine peut faire — et donc ce qu'elle
pourrait casser. Ces tests portent d'abord sur les refus.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from capucine.core import plugin as contrat
from capucine.core.atelier import Atelier
from capucine.core.config import PROJECT_ROOT, Config
from capucine.core.conversation import Conversation
from capucine.core.memoire import Memoire
from capucine.core.registry import PluginRegistry

DOSSIER = PROJECT_ROOT / "plugins"


@pytest.fixture
def banc(tmp_path: Path):
    """Un registre chargé des plugins livrés, avec des ressources maîtrisées."""
    travail = tmp_path / "projet"
    travail.mkdir()
    annonces: list[str] = []
    reponses: list[str] = []

    espace = Atelier(racines=[travail], corbeille=tmp_path / "corbeille")
    magasin = Memoire(tmp_path / "memoire.sqlite")
    fil = Conversation(persona="persona", memoire=magasin)
    fil.session_id = magasin.ouvrir_session().id

    contrat.set_announcer(annonces.append)
    contrat.set_atelier(espace)
    contrat.set_memoire(magasin)
    contrat.set_conversation(fil)
    contrat.set_model_access(lambda p, **k: reponses.pop(0) if reponses else "")

    def _registre(config: dict | None = None) -> PluginRegistry:
        registry = PluginRegistry(
            [DOSSIER], config=Config(config or {}), data_root=tmp_path / "data"
        )
        registry.load_all()
        return registry

    yield {
        "registre": _registre, "travail": travail, "annonces": annonces,
        "reponses": reponses, "atelier": espace, "memoire": magasin, "fil": fil,
    }

    magasin.fermer()
    for sink in (contrat.set_announcer, contrat.set_atelier,
                 contrat.set_memoire, contrat.set_conversation, contrat.set_model_access):
        sink(None)


# --- recherche --------------------------------------------------------------

def test_un_moteur_inconnu_est_annonce(banc) -> None:
    registry = banc["registre"]({"plugins": {"recherche": {"moteur": "bing"}}})
    resultat = registry.call("chercher", {"question": "test"})
    assert "ne m'est pas connu" in resultat.speak
    assert "searxng" in resultat.display


def test_google_sans_cle_dit_ce_qu_il_manque(banc) -> None:
    registry = banc["registre"]({"plugins": {"recherche": {"moteur": "google"}}})
    resultat = registry.call("chercher", {"question": "test"})
    assert "clé d'API" in resultat.speak
    assert "google_cle_api" in resultat.speak


def test_un_moteur_injoignable_ne_plante_pas(banc) -> None:
    # Wi-Fi coupé, ou instance SearXNG éteinte : elle le dit, elle ne casse pas.
    registry = banc["registre"]({
        "plugins": {"recherche": {"moteur": "searxng", "searxng_url": "http://127.0.0.1:1", "delai_s": 1.0}}
    })
    resultat = registry.call("chercher", {"question": "test"})
    assert resultat.ok
    assert "n'arrive pas à joindre" in resultat.speak


def test_l_extraction_de_texte_d_une_page(banc) -> None:
    from plugins.recherche import _Texte

    analyseur = _Texte()
    analyseur.feed(
        "<html><head><style>p{color:red}</style></head>"
        "<body><script>var x=1</script><h1>Titre</h1>"
        "<p>Premier paragraphe.</p><p>Second paragraphe.</p></body></html>"
    )
    texte = analyseur.texte()
    assert "Titre" in texte and "Premier paragraphe." in texte
    # Le script et le style ne sont pas du contenu.
    assert "var x" not in texte and "color:red" not in texte


def test_les_liens_duckduckgo_sont_deplies() -> None:
    from plugins.recherche import _url_reelle

    enrobe = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexemple.fr%2Fpage&rut=abc"
    assert _url_reelle(enrobe) == "https://exemple.fr/page"
    assert _url_reelle("https://direct.fr") == "https://direct.fr"


# --- fichiers ---------------------------------------------------------------

def test_un_refus_de_l_atelier_est_prononce(banc) -> None:
    registry = banc["registre"]()
    resultat = registry.call("lire_fichier", {"chemin": "/etc/passwd"})
    # Pas de « je n'ai pas pu exécuter cette commande » : la raison exacte.
    assert "hors de l'atelier" in resultat.speak
    # Et ce n'est pas compté comme une panne du plugin.
    assert registry.get("lire_fichier").failures == 0


def test_lister_et_chercher_dans_les_fichiers(banc) -> None:
    (banc["travail"] / "calcul.py").write_text(
        "def dcf():\n    return 42\n", encoding="utf-8"
    )
    registry = banc["registre"]()
    assert "calcul.py" in registry.call("lister_fichiers").speak
    trouvaille = registry.call("chercher_dans_fichiers", {"texte": "dcf"})
    assert "calcul.py" in trouvaille.display


def test_completer_un_fichier_ne_demande_pas_confirmation(banc) -> None:
    registry = banc["registre"]()
    resultat = registry.call("ajouter_au_fichier", {"chemin": "notes.txt", "contenu": "une ligne"})
    assert resultat.ok and not resultat.needs_confirmation
    assert (banc["travail"] / "notes.txt").read_text(encoding="utf-8") == "une ligne\n"


@pytest.mark.parametrize(
    "competence,arguments",
    [
        ("ecrire_fichier", {"chemin": "notes.txt", "contenu": "écrasé"}),
        ("jeter_fichier", {"chemin": "notes.txt"}),
        ("deplacer_fichier", {"source": "notes.txt", "destination": "ailleurs.txt"}),
    ],
)
def test_tout_ce_qui_detruit_demande_confirmation(banc, competence, arguments) -> None:
    (banc["travail"] / "notes.txt").write_text("original", encoding="utf-8")
    registry = banc["registre"]()

    demande = registry.call(competence, arguments)
    assert demande.needs_confirmation
    assert (banc["travail"] / "notes.txt").read_text(encoding="utf-8") == "original"


def test_ecrire_garde_une_sauvegarde(banc) -> None:
    (banc["travail"] / "notes.txt").write_text("original", encoding="utf-8")
    registry = banc["registre"]()
    registry.call("ecrire_fichier", {"chemin": "notes.txt", "contenu": "nouveau"}, confirmed=True)

    assert (banc["travail"] / "notes.txt").read_text(encoding="utf-8") == "nouveau"
    assert list(banc["travail"].glob("notes.txt.*.sauvegarde"))


def test_l_atelier_se_decrit(banc) -> None:
    registry = banc["registre"]()
    assert "lecture et écriture" in registry.call("mon_atelier").speak


# --- python -----------------------------------------------------------------

def test_executer_du_python_rend_sa_sortie(banc) -> None:
    registry = banc["registre"]()
    resultat = registry.call("executer_python", {"code": "print(6*7)"}, confirmed=True)
    assert "42" in resultat.speak
    assert "code de retour 0" in resultat.display


def test_un_code_qui_leve_est_rapporte_sans_planter(banc) -> None:
    registry = banc["registre"]()
    resultat = registry.call("executer_python", {"code": "raise ValueError('boum')"}, confirmed=True)
    assert resultat.ok            # la compétence a fonctionné
    assert "ValueError" in resultat.speak   # c'est le code qui a échoué


def test_executer_du_python_demande_confirmation(banc) -> None:
    registry = banc["registre"]()
    assert registry.call("executer_python", {"code": "print(1)"}).needs_confirmation


def test_le_code_propose_est_montre_avant_d_etre_ecrit(banc) -> None:
    banc["reponses"].append("```python\ndef moyenne(v):\n    return sum(v)/len(v)\n```")
    registry = banc["registre"]()

    proposition = registry.call("ecrire_du_code", {"description": "une moyenne"})
    # Le code est visible, et rien n'a touché le disque.
    assert "def moyenne" in proposition.display
    assert "```" not in proposition.display      # les balises Markdown sont retirées
    assert not list(banc["travail"].glob("*.py"))

    demande = registry.call("enregistrer_le_code", {"chemin": "moyenne.py"})
    assert demande.needs_confirmation

    registry.call("enregistrer_le_code", {"chemin": "moyenne.py"}, confirmed=True)
    assert "def moyenne" in (banc["travail"] / "moyenne.py").read_text(encoding="utf-8")


def test_enregistrer_sans_proposition_le_dit(banc) -> None:
    registry = banc["registre"]()
    resultat = registry.call("enregistrer_le_code", {"chemin": "vide.py"}, confirmed=True)
    assert "aucun code en attente" in resultat.speak.lower()


def test_lancer_un_script_hors_de_l_atelier_est_refuse(banc) -> None:
    registry = banc["registre"]()
    resultat = registry.call("lancer_script", {"chemin": "/etc/hosts"}, confirmed=True)
    assert "hors de l'atelier" in resultat.speak


def test_un_fichier_non_python_n_est_pas_lance(banc) -> None:
    (banc["travail"] / "notes.txt").write_text("x", encoding="utf-8")
    registry = banc["registre"]()
    resultat = registry.call("lancer_script", {"chemin": "notes.txt"}, confirmed=True)
    assert "n'est pas un script Python" in resultat.speak


# --- projet -----------------------------------------------------------------

def _config_projet(dossier: Path) -> dict:
    return {"plugins": {"projet": {"projets": {"essai": {
        "chemin": str(dossier),
        "description": "pipeline de démonstration",
        "commande": "pipeline.py",
        "rapport": "runs/*.json",
        "commande_test": '-c \'print("tout va bien")\'',
    }}}}}


def test_sans_projet_declare_on_le_dit(banc) -> None:
    registry = banc["registre"]()
    assert "Aucun projet" in registry.call("mes_projets").speak
    assert "Aucun projet" in registry.call("lancer_projet", {}, confirmed=True).speak


def test_un_projet_inconnu_liste_ceux_qui_existent(banc) -> None:
    registry = banc["registre"](_config_projet(banc["travail"]))
    resultat = registry.call("lancer_projet", {"nom": "zzz"}, confirmed=True)
    assert "essai" in resultat.speak


def test_un_projet_tourne_en_fond_et_previent_a_la_fin(banc) -> None:
    (banc["travail"] / "pipeline.py").write_text(
        'import time\nprint("étape 1")\ntime.sleep(0.3)\nprint("fini")\n', encoding="utf-8"
    )
    registry = banc["registre"](_config_projet(banc["travail"]))

    depart = registry.call("lancer_projet", {"nom": "essai"}, confirmed=True)
    assert "C'est parti" in depart.speak      # la main est rendue tout de suite

    limite = time.monotonic() + 15
    while not banc["annonces"] and time.monotonic() < limite:
        time.sleep(0.05)
    assert banc["annonces"] and "terminé" in banc["annonces"][0]


def test_un_projet_en_echec_est_annonce_avec_sa_derniere_ligne(banc) -> None:
    (banc["travail"] / "pipeline.py").write_text(
        'import sys\nprint("plantage attendu")\nsys.exit(3)\n', encoding="utf-8"
    )
    registry = banc["registre"](_config_projet(banc["travail"]))
    registry.call("lancer_projet", {"nom": "essai"}, confirmed=True)

    limite = time.monotonic() + 15
    while not banc["annonces"] and time.monotonic() < limite:
        time.sleep(0.05)
    assert banc["annonces"] and "erreur" in banc["annonces"][0]


def test_l_etat_lit_le_rapport_du_dernier_run(banc) -> None:
    (banc["travail"] / "runs").mkdir()
    (banc["travail"] / "runs" / "report.json").write_text(
        json.dumps({
            "status": "partial", "duration_seconds": 300,
            "steps": [{"script": "05.py", "status": "success"},
                      {"script": "08.py", "status": "failed"}],
        }), encoding="utf-8",
    )
    registry = banc["registre"](_config_projet(banc["travail"]))
    resultat = registry.call("etat_projet", {"nom": "essai"})
    assert "partial" in resultat.speak
    assert "08.py" in resultat.speak


def test_un_projet_hors_de_l_atelier_est_refuse(banc, tmp_path: Path) -> None:
    ailleurs = tmp_path / "ailleurs"
    ailleurs.mkdir()
    registry = banc["registre"](_config_projet(ailleurs))
    resultat = registry.call("etat_projet", {"nom": "essai"})
    # Le projet doit se trouver dans un dossier ouvert : sinon, rien à dire.
    assert "ne tourne pas" in resultat.speak


def test_les_tests_d_un_projet_sont_rapportes(banc) -> None:
    # La commande contient des guillemets : elle doit être découpée par shlex,
    # pas par un split() naïf qui la couperait au milieu.
    registry = banc["registre"](_config_projet(banc["travail"]))
    resultat = registry.call("tester_projet", {"nom": "essai"}, confirmed=True)
    assert "Les tests passent" in resultat.speak
    assert "tout va bien" in resultat.display


def test_les_arguments_entre_guillemets_restent_entiers(banc) -> None:
    (banc["travail"] / "montre.py").write_text(
        "import sys\nprint(sys.argv[1:])\n", encoding="utf-8"
    )
    registry = banc["registre"]()
    resultat = registry.call(
        "lancer_script",
        {"chemin": "montre.py", "arguments": '--titre "deux mots" --n 3'},
        confirmed=True,
    )
    assert "'deux mots'" in resultat.display


# --- mémoire, vue depuis les compétences ------------------------------------

def test_retenir_puis_relire(banc) -> None:
    registry = banc["registre"]()
    assert registry.call("retenir", {"fait": "Je m'appelle Tom"}).speak == "C'est retenu."
    assert "Tom" in registry.call("ce_que_tu_sais").speak
    assert registry.call("retenir", {"fait": "Je m'appelle Tom"}).speak == "Je le savais déjà."


def test_oublier_demande_confirmation(banc) -> None:
    registry = banc["registre"]()
    registry.call("retenir", {"fait": "Je m'appelle Tom"})
    assert registry.call("oublier", {"sujet": "Tom"}).needs_confirmation
    assert "Tom" in registry.call("ce_que_tu_sais").speak

    registry.call("oublier", {"sujet": "Tom"}, confirmed=True)
    assert "rien de particulier" in registry.call("ce_que_tu_sais").speak


def test_retrouver_et_reprendre_une_conversation(banc) -> None:
    magasin, fil = banc["memoire"], banc["fil"]
    ancienne = magasin.ouvrir_session()
    magasin.ajouter_message(ancienne.id, "user", "parlons du backtest options")
    magasin.ajouter_message(ancienne.id, "assistant", "avec plaisir")

    registry = banc["registre"]()
    assert "backtest" in registry.call("retrouver", {"sujet": "backtest"}).speak
    assert "0 résultat" in registry.call("retrouver", {"sujet": "licorne"}).display

    reprise = registry.call("reprendre_conversation", {"numero": ancienne.id})
    assert "reprenons" in reprise.speak
    assert fil.session_id == ancienne.id
    assert [m.content for m in fil.history()] == ["parlons du backtest options", "avec plaisir"]

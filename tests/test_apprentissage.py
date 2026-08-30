"""Ce que Capucine apprend de vous : formulations, corrections, vocabulaire."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from capucine.core.apprentissage import Apprentissage, depuis_config
from capucine.core.config import Config
from capucine.core.conversation import Conversation
from capucine.core.engines.llm.mock import MockLLM
from capucine.core.pipeline import Pipeline
from capucine.core.plugin import SkillDeclaration, build_skill_spec
from capucine.core.registry import PluginRegistry
from capucine.core.router import NO_TOOL, Router
from capucine.core.text import est_une_correction


@pytest.fixture
def magasin(tmp_path: Path) -> Apprentissage:
    instance = Apprentissage(tmp_path / "appris.sqlite")
    yield instance
    instance.fermer()


# --- 1. apprendre les formulations ------------------------------------------

def test_une_formulation_retenue_prend_du_poids(magasin: Apprentissage) -> None:
    magasin.apprendre_routage("relance le calcul du backtest", "lancer_projet")
    (retenue,) = magasin.phrases_par_outil()["lancer_projet"]
    assert retenue.confirmations == 1
    assert retenue.poids == pytest.approx(0.8)

    magasin.apprendre_routage("relance le calcul du backtest", "lancer_projet")
    (retenue,) = magasin.phrases_par_outil()["lancer_projet"]
    assert retenue.confirmations == 2
    # Une formulation confirmée pèse presque autant qu'un exemple d'auteur,
    # jamais davantage.
    assert 0.8 < retenue.poids <= 1.0


def test_une_phrase_trop_courte_n_est_pas_retenue(magasin: Apprentissage) -> None:
    assert not magasin.apprendre_routage("oui", "heure")
    assert magasin.phrases_par_outil() == {}


def test_le_nombre_de_phrases_par_competence_est_borne(tmp_path: Path) -> None:
    magasin = Apprentissage(tmp_path / "a.sqlite", phrases_par_competence=3)
    try:
        for numero in range(10):
            magasin.apprendre_routage(f"formulation numero {numero}", "heure")
        assert len(magasin.phrases_par_outil()["heure"]) == 3
    finally:
        magasin.fermer()


def test_un_dementi_finit_par_effacer(magasin: Apprentissage) -> None:
    magasin.apprendre_routage("mets-moi trois minutes", "lancer_projet")
    magasin.dementir_routage("mets-moi trois minutes", "lancer_projet")
    # Une erreur apprise doit se désapprendre aussi vite qu'elle s'est apprise.
    assert magasin.phrases_par_outil() == {}


def test_une_formulation_bien_ancree_resiste_a_un_dementi(magasin: Apprentissage) -> None:
    for _ in range(3):
        magasin.apprendre_routage("relance le pipeline complet", "lancer_projet")
    magasin.dementir_routage("relance le pipeline complet", "lancer_projet")
    (retenue,) = magasin.phrases_par_outil()["lancer_projet"]
    assert retenue.dementis == 1 and retenue.confirmations == 3


def test_oublier_une_competence_disparue(magasin: Apprentissage) -> None:
    magasin.apprendre_routage("une formulation quelconque", "outil_disparu")
    assert magasin.oublier_outil("outil_disparu") == 1
    assert magasin.phrases_par_outil() == {}


# --- le routeur s'en sert ----------------------------------------------------

def _spec(fonction, description, exemples):
    return build_skill_spec(
        fonction,
        SkillDeclaration(description, tuple(exemples), None, None, False, False),
        module="capucine.plugins.essai", source=Path("essai.py"), plugin="essai",
    )


def heure() -> str:
    """Donne l'heure."""
    return "midi"


def lancer_projet(nom: str = "") -> str:
    """Lance un projet."""
    return "parti"


def test_une_formulation_apprise_evite_le_modele(magasin: Apprentissage) -> None:
    skills = {
        "heure": _spec(heure, "Donne l'heure.", ["quelle heure est-il"]),
        "lancer_projet": _spec(lancer_projet, "Lance un projet.", ["lance le projet"]),
    }
    phrase = "relance-moi le calcul complet du bazar"

    # Sans apprentissage, l'étage déterministe ne reconnaît pas : le modèle tranche.
    llm = MockLLM([json.dumps({"outil": "lancer_projet"}), json.dumps({})])
    decision = Router(llm).route(phrase, skills)
    assert decision.tier == "llm"

    # Une fois la formulation retenue, l'étage déterministe suffit.
    magasin.apprendre_routage(phrase, "lancer_projet")
    llm2 = MockLLM()
    decision = Router(llm2, apprentissage=magasin).route(phrase, skills)
    assert decision.tier == "regle"
    assert decision.tool_call.name == "lancer_projet"
    assert llm2.calls == []      # aucune inférence


def test_le_routeur_sans_apprentissage_fonctionne_toujours() -> None:
    skills = {"heure": _spec(heure, "Donne l'heure.", ["quelle heure est-il"])}
    decision = Router(MockLLM(), apprentissage=None).route("quelle heure est-il", skills)
    assert decision.tier == "regle"


# --- 2. apprendre des corrections -------------------------------------------

@pytest.mark.parametrize(
    "phrase,attendu",
    [
        ("non je voulais dire le minuteur", True),
        ("plutôt trois minutes", True),
        ("en fait mets une alarme", True),
        ("non", False),                       # un refus sec, traité ailleurs
        ("quelle heure est-il", False),
        ("", False),
    ],
)
def test_reconnaitre_une_correction(phrase, attendu) -> None:
    assert est_une_correction(phrase) is attendu


PLUGIN = '''
from capucine.plugin import skill

@skill(description="Donne l'heure.", examples=["quelle heure est-il"])
def heure() -> str:
    return "Il est midi."

@skill(description="Lance un minuteur.", examples=["mets un minuteur"])
def minuteur(minutes: int = 5) -> str:
    return "minuteur lancé"
'''


def _pipeline(dossier: Path, magasin: Apprentissage, llm: MockLLM) -> Pipeline:
    registry = PluginRegistry([dossier], data_root=dossier.parent / "data")
    registry.load_all()
    return Pipeline(
        registry,
        Router(llm, apprentissage=magasin),
        Conversation(persona="persona", max_turns=4),
        apprentissage=magasin,
    )


def test_un_tour_route_par_le_modele_est_retenu(
    ecrire_plugin, dossier_plugins, magasin
) -> None:
    ecrire_plugin("essai.py", PLUGIN)
    llm = MockLLM([json.dumps({"outil": "minuteur"}), json.dumps({"minutes": 3})])
    pipeline = _pipeline(dossier_plugins, magasin, llm)

    phrase = "programme-moi un rappel dans trois minutes"
    asyncio.run(pipeline.handle(phrase))

    assert "minuteur" in magasin.phrases_par_outil()
    assert magasin.phrases_par_outil()["minuteur"][0].phrase == phrase


def test_un_tour_deja_deterministe_n_apprend_rien(
    ecrire_plugin, dossier_plugins, magasin
) -> None:
    # Inutile de retenir ce que l'étage déterministe savait déjà faire.
    ecrire_plugin("essai.py", PLUGIN)
    pipeline = _pipeline(dossier_plugins, magasin, MockLLM())
    asyncio.run(pipeline.handle("quelle heure est-il"))
    assert magasin.phrases_par_outil() == {}


def test_une_correction_desapprend_et_reapprend(
    ecrire_plugin, dossier_plugins, magasin
) -> None:
    ecrire_plugin("essai.py", PLUGIN)
    # Premier tour : le modèle se trompe et route vers l'heure. Une seule
    # réponse pour ce tour-là : « heure » n'a pas de paramètre, la seconde
    # passe contrainte n'a rien à extraire et le modèle n'est pas sollicité.
    llm = MockLLM([
        json.dumps({"outil": "heure"}),
        json.dumps({"outil": "minuteur"}), json.dumps({"minutes": 3}),
    ])
    pipeline = _pipeline(dossier_plugins, magasin, llm)

    phrase = "programme-moi un rappel dans trois minutes"

    async def scenario() -> None:
        await pipeline.handle(phrase)
        await pipeline.handle("non je voulais dire le minuteur")

    asyncio.run(scenario())

    table = magasin.phrases_par_outil()
    # La phrase d'origine a changé de camp : c'est elle qui reviendra, pas
    # la correction.
    assert "heure" not in table
    assert table["minuteur"][0].phrase == phrase


def test_une_correction_sans_tour_precedent_ne_casse_rien(
    ecrire_plugin, dossier_plugins, magasin
) -> None:
    ecrire_plugin("essai.py", PLUGIN)
    llm = MockLLM([json.dumps({"outil": NO_TOOL}), "d'accord"])
    pipeline = _pipeline(dossier_plugins, magasin, llm)
    resultat = asyncio.run(pipeline.handle("non je voulais dire autre chose"))
    assert resultat.tier == "conversation"


def test_un_apprentissage_en_panne_ne_casse_pas_un_tour(
    ecrire_plugin, dossier_plugins
) -> None:
    class MagasinCasse:
        corrections_actives = True

        def moissonner(self, *args, **kwargs):
            raise OSError("disque plein")

        def phrases_par_outil(self):
            return {}

    ecrire_plugin("essai.py", PLUGIN)
    pipeline = _pipeline(dossier_plugins, MagasinCasse(), MockLLM())
    resultat = asyncio.run(pipeline.handle("quelle heure est-il"))
    assert resultat.speak == "Il est midi."      # le tour a abouti


# --- 3. le vocabulaire -------------------------------------------------------

@pytest.mark.parametrize(
    "texte,attendus",
    [
        ("Le pipeline CalculRisque_Mark5 lit les 10K", ["CalculRisque_Mark5", "10K"]),
        ("On regarde l'EBITDA et le modèle qwen2", ["EBITDA", "qwen2"]),
        ("Bonjour, quelle heure est-il ?", []),
        ("Mon projet ValoOptions avance bien", ["ValoOptions"]),
    ],
)
def test_moissonner_ce_que_whisper_ecorche(magasin, texte, attendus) -> None:
    # Volontairement conservateur : un faux positif dans l'amorce biaise la
    # transcription vers des mots que vous ne dites jamais.
    assert magasin.moissonner(texte) == attendus


def test_un_mot_n_est_moissonne_qu_une_fois(magasin: Apprentissage) -> None:
    assert magasin.moissonner("EBITDA et encore EBITDA") == ["EBITDA"]
    assert magasin.moissonner("toujours EBITDA") == []
    (entree,) = magasin.vocabulaire()
    assert entree.occurrences == 3


def test_l_amorce_de_transcription_contient_le_vocabulaire(magasin: Apprentissage) -> None:
    magasin.moissonner("Le dépôt CalculRisque_Mark5 et la SEC")
    amorce = magasin.amorce_stt("Capucine, assistante vocale.")
    assert amorce.startswith("Capucine, assistante vocale.")
    assert "CalculRisque_Mark5" in amorce and "SEC" in amorce


def test_le_vocabulaire_est_souffle_a_la_transcription(
    ecrire_plugin, dossier_plugins, magasin
) -> None:
    from capucine.core.audio import AudioBuffer
    from capucine.core.engines.stt.scripted import ScriptedSTT
    from capucine.core.logging import TurnTelemetry

    class STTAmorcable(ScriptedSTT):
        initial_prompt = "Capucine, assistante vocale."

    ecrire_plugin("essai.py", PLUGIN)
    stt = STTAmorcable(["quelle heure est-il"])
    registry = PluginRegistry([dossier_plugins], data_root=dossier_plugins.parent / "d")
    registry.load_all()
    pipeline = Pipeline(
        registry, Router(MockLLM(), apprentissage=magasin),
        Conversation(persona="p"), stt=stt, apprentissage=magasin,
    )
    magasin.moissonner("Le dépôt CalculRisque_Mark5")

    asyncio.run(pipeline.transcribe(AudioBuffer(b"\x00\x00" * 8000, 16000), TurnTelemetry()))
    assert "CalculRisque_Mark5" in stt.initial_prompt
    assert stt.initial_prompt.startswith("Capucine, assistante vocale.")


def test_un_mot_courant_n_encombre_pas_le_vocabulaire(magasin: Apprentissage) -> None:
    assert not magasin.retenir_mot("Capucine")
    assert not magasin.retenir_mot("ok")


def test_oublier_tout(magasin: Apprentissage) -> None:
    magasin.apprendre_routage("une formulation quelconque", "heure")
    magasin.moissonner("EBITDA")
    assert magasin.statistiques() == {"phrases": 1, "competences": 1, "mots": 1}
    magasin.tout_oublier()
    assert magasin.statistiques() == {"phrases": 0, "competences": 0, "mots": 0}


# --- configuration -----------------------------------------------------------

def test_l_apprentissage_peut_etre_desactive(tmp_path: Path) -> None:
    config = Config({"apprentissage": {"active": False}})
    assert depuis_config(config) is None


def test_chaque_mecanisme_se_desactive_separement(tmp_path: Path) -> None:
    config = Config({
        "apprentissage": {"routage": False, "vocabulaire": False,
                          "fichier": str(tmp_path / "a.sqlite")},
    })
    magasin = depuis_config(config)
    try:
        assert not magasin.apprendre_routage("une phrase assez longue", "heure")
        assert magasin.moissonner("EBITDA") == []
        assert magasin.amorce_stt("base") == "base"
    finally:
        magasin.fermer()

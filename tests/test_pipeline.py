"""Machine à états, avec des composants factices.

Aucun modèle, aucun micro, aucun haut-parleur : le tour complet est vérifiable
sur une machine nue.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

from capucine.app import build_assistant
from capucine.core.config import Config
from capucine.core.conversation import Conversation, load_persona
from capucine.core.engines.llm.mock import MockLLM
from capucine.core.pipeline import Pipeline, State
from capucine.core.registry import PluginRegistry
from capucine.core.router import NO_TOOL, Router

PLUGIN_HEURE = '''
from capucine.plugin import skill

@skill(description="Donne l'heure.", examples=["quelle heure est-il"])
def heure() -> str:
    """Donne l'heure courante."""
    return "Il est midi."
'''


def monter(dossier, llm=None, **kwargs):
    """Assemble un pipeline complet autour d'un dossier de plugins."""
    llm = llm or MockLLM()
    registry = PluginRegistry([dossier], data_root=dossier.parent / "data")
    registry.load_all()
    pipeline = Pipeline(
        registry, Router(llm), Conversation(persona="persona de test", max_turns=3), **kwargs
    )
    return pipeline, registry, llm


def test_un_tour_avec_outil(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    pipeline, _, llm = monter(dossier_plugins)

    resultat = asyncio.run(pipeline.handle("quelle heure est-il"))

    assert resultat.tier == "regle"
    assert resultat.tool.name == "heure"
    assert resultat.speak == "Il est midi."
    assert llm.calls == []
    # Les latences par étage sont mesurées, c'est ce qui servira à profiler le Pi.
    assert "reflexion_ms" in resultat.telemetry.stages
    assert "execution_ms" in resultat.telemetry.stages


def test_un_tour_sans_outil_passe_par_la_conversation(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    llm = MockLLM([json.dumps({"outil": NO_TOOL}), "Bonsoir."])
    pipeline, _, _ = monter(dossier_plugins, llm)

    resultat = asyncio.run(pipeline.handle("raconte-moi ta journée"))

    assert resultat.tier == "conversation"
    assert resultat.speak == "Bonsoir."
    assert "reponse_ms" in resultat.telemetry.stages


def test_un_plugin_en_echec_donne_une_reponse_pas_une_exception(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("casse.py", '''
from capucine.plugin import skill

@skill(description="Explose.", examples=["fais tout planter"])
def exploser() -> str:
    raise RuntimeError("boum")
''')
    pipeline, _, _ = monter(dossier_plugins)
    resultat = asyncio.run(pipeline.handle("fais tout planter"))

    assert resultat.skill_result is not None and not resultat.skill_result.ok
    assert resultat.speak == "Je n'ai pas pu exécuter cette commande."


def test_les_etats_sont_traverses_dans_l_ordre(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    etats: list[State] = []
    pipeline, _, _ = monter(dossier_plugins, on_state=etats.append)

    asyncio.run(pipeline.handle_and_speak("quelle heure est-il"))

    assert etats == [State.THINK, State.ACT, State.SPEAK, State.IDLE]
    assert pipeline.state is State.IDLE


def test_la_memoire_courte_est_bornee(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    pipeline, _, _ = monter(dossier_plugins)

    async def scenario() -> None:
        for _ in range(6):
            await pipeline.handle("quelle heure est-il")

    asyncio.run(scenario())
    # max_turns=3 -> six messages au plus (question + réponse par tour).
    assert len(pipeline.conversation) == 6


def test_un_tour_peut_etre_annule_c_est_le_barge_in(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("lent.py", '''
import time
from capucine.plugin import skill

@skill(description="Prend son temps.", examples=["prends ton temps"], timeout=5)
def lambiner() -> str:
    time.sleep(2)
    return "enfin"
''')
    pipeline, _, _ = monter(dossier_plugins)

    async def scenario() -> bool:
        pipeline.attach()
        tache = pipeline.start_turn("prends ton temps")
        await asyncio.sleep(0.05)
        annule = pipeline.cancel_turn()
        with contextlib.suppress(asyncio.CancelledError):
            await tache
        return annule

    assert asyncio.run(scenario()) is True


def test_un_plugin_peut_interrompre_capucine_depuis_une_tache_de_fond(
    ecrire_plugin, dossier_plugins
) -> None:
    # C'est le mécanisme dont aura besoin le minuteur : la tâche de fond n'a
    # personne à qui répondre, elle doit pouvoir interrompre.
    ecrire_plugin("sonnette.py", '''
import threading
from capucine.plugin import announce, skill

@skill(description="Sonne plus tard.", examples=["sonne dans un instant"])
def sonner() -> str:
    threading.Timer(0.05, lambda: announce("La minuterie est écoulée.")).start()
    return "C'est parti."
''')
    dits: list[str] = []
    pipeline, _, _ = monter(dossier_plugins, speak=dits.append)

    async def scenario() -> None:
        pipeline.attach()
        try:
            await pipeline.handle_and_speak("sonne dans un instant")
            await asyncio.sleep(0.2)
            await pipeline.drain_announcements()
        finally:
            pipeline.detach()

    asyncio.run(scenario())
    assert dits == ["C'est parti.", "La minuterie est écoulée."]


def test_une_nouvelle_competence_est_annoncee_une_fois(ecrire_plugin, dossier_plugins) -> None:
    pipeline, registry, _ = monter(dossier_plugins)
    registry.on_change = pipeline.notify_skill_change
    dits: list[str] = []
    pipeline._speak = dits.append

    async def scenario() -> None:
        pipeline.attach()
        try:
            chemin = ecrire_plugin("horloge.py", PLUGIN_HEURE)
            registry.load_file(chemin)
            await pipeline.drain_announcements()
            # Une simple modification, sans nouveau nom, reste silencieuse.
            registry.reload_file(chemin)
            await pipeline.drain_announcements()
        finally:
            pipeline.detach()

    asyncio.run(scenario())
    assert dits == ["Nouvelle compétence disponible : heure."]


def test_une_panne_de_routage_reste_une_phrase(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)

    class LLMQuiCasse(MockLLM):
        def chat(self, *args, **kwargs):
            raise OSError("moteur injoignable")

    pipeline, _, _ = monter(dossier_plugins, LLMQuiCasse())
    resultat = asyncio.run(pipeline.handle("raconte-moi quelque chose de long"))

    assert resultat.tier == "erreur"
    assert resultat.speak == "Je n'ai pas réussi à traiter cette demande."


def test_une_phrase_vide_ne_fait_rien(dossier_plugins) -> None:
    pipeline, _, _ = monter(dossier_plugins)
    resultat = asyncio.run(pipeline.handle("   "))
    assert resultat.speak == ""
    assert len(pipeline.conversation) == 0


def test_l_assistant_complet_se_monte_depuis_une_config(tmp_path) -> None:
    # Chemin réel de build_assistant, avec les plugins livrés.
    config = Config({
        "profile": "pc",
        "assistant": {"memory_turns": 4, "persona_file": "config/persona.txt"},
        "plugins": {"paths": ["./plugins"], "timeout": 5.0, "data_dir": str(tmp_path / "data")},
        "llm": {"engine": "mock", "router": {"direct_threshold": 0.72}},
    })
    assistant = build_assistant(config, llm=MockLLM())
    try:
        # Les quatre plugins livrés sont chargés et leurs compétences prêtes.
        assert {"heure", "minuteur", "noter", "etat_systeme"} <= set(assistant.registry.skills)
        assert not assistant.registry.failures()
        resultat = asyncio.run(assistant.pipeline.handle("quelle heure est-il"))
        assert resultat.tier == "regle"
        assert resultat.speak.startswith("Il est ")
    finally:
        asyncio.run(assistant.aclose())


def test_le_persona_livre_est_lisible() -> None:
    from capucine.core.config import PROJECT_ROOT

    persona = load_persona(PROJECT_ROOT / "config" / "persona.txt")
    assert "Capucine" in persona
    persona_absent = load_persona(PROJECT_ROOT / "config" / "inexistant.txt")
    assert "Capucine" in persona_absent   # repli, jamais de plantage

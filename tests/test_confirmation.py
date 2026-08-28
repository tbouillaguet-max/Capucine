"""Actions irréversibles : Capucine demande avant de faire."""

from __future__ import annotations

import asyncio

from capucine.core.conversation import Conversation
from capucine.core.engines.llm.mock import MockLLM
from capucine.core.pipeline import Pipeline
from capucine.core.registry import PluginRegistry
from capucine.core.router import Router
from capucine.core.text import accord_ou_refus

PLUGIN_DESTRUCTEUR = '''
from capucine.plugin import skill

EFFACEMENTS = []

@skill(
    description="Efface tout.",
    examples=["efface tout"],
    confirm="Voulez-vous vraiment tout effacer ?",
)
def effacer() -> str:
    EFFACEMENTS.append(1)
    return "Tout est effacé."

@skill(description="Compte les effacements.", examples=["combien d'effacements"])
def compter() -> str:
    return str(len(EFFACEMENTS))
'''


def monter(dossier, llm=None):
    registry = PluginRegistry([dossier], data_root=dossier.parent / "data")
    registry.load_all()
    pipeline = Pipeline(
        registry, Router(llm or MockLLM()),
        Conversation(persona="persona de test", max_turns=4),
    )
    return pipeline, registry


def test_une_action_irreversible_pose_une_question(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("destructeur.py", PLUGIN_DESTRUCTEUR)
    pipeline, registry = monter(dossier_plugins)

    resultat = asyncio.run(pipeline.handle("efface tout"))

    assert resultat.speak == "Voulez-vous vraiment tout effacer ?"
    assert resultat.skill_result.needs_confirmation
    # Rien n'a été exécuté.
    assert registry.call("compter").speak == "0"


def test_un_oui_execute(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("destructeur.py", PLUGIN_DESTRUCTEUR)
    pipeline, registry = monter(dossier_plugins)

    async def scenario() -> str:
        await pipeline.handle("efface tout")
        return (await pipeline.handle("oui")).speak

    assert asyncio.run(scenario()) == "Tout est effacé."
    assert registry.call("compter").speak == "1"


def test_un_non_annule(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("destructeur.py", PLUGIN_DESTRUCTEUR)
    pipeline, registry = monter(dossier_plugins)

    async def scenario() -> str:
        await pipeline.handle("efface tout")
        return (await pipeline.handle("non")).speak

    assert asyncio.run(scenario()) == "Très bien, je n'ai rien fait."
    assert registry.call("compter").speak == "0"


def test_une_reponse_qui_ne_tranche_pas_annule_l_attente(
    ecrire_plugin, dossier_plugins
) -> None:
    # On ne piège pas l'utilisateur dans une question : s'il parle d'autre
    # chose, on abandonne l'attente et on traite sa nouvelle demande.
    ecrire_plugin("destructeur.py", PLUGIN_DESTRUCTEUR)
    pipeline, registry = monter(dossier_plugins)

    async def scenario() -> str:
        await pipeline.handle("efface tout")
        return (await pipeline.handle("combien d'effacements")).speak

    assert asyncio.run(scenario()) == "0"
    assert pipeline._confirmation is None
    assert registry.call("compter").speak == "0"


def test_un_oui_sans_question_en_attente_ne_declenche_rien(
    ecrire_plugin, dossier_plugins
) -> None:
    ecrire_plugin("destructeur.py", PLUGIN_DESTRUCTEUR)
    pipeline, registry = monter(dossier_plugins)

    asyncio.run(pipeline.handle("oui"))
    assert registry.call("compter").speak == "0"


def test_la_confirmation_survit_a_une_reponse_bavarde(
    ecrire_plugin, dossier_plugins
) -> None:
    ecrire_plugin("destructeur.py", PLUGIN_DESTRUCTEUR)
    pipeline, registry = monter(dossier_plugins)

    async def scenario() -> str:
        await pipeline.handle("efface tout")
        return (await pipeline.handle("oui vas-y")).speak

    assert asyncio.run(scenario()) == "Tout est effacé."
    assert registry.call("compter").speak == "1"


def test_lecture_des_accords_et_des_refus() -> None:
    assert accord_ou_refus("oui") is True
    assert accord_ou_refus("vas-y") is True
    assert accord_ou_refus("non merci") is False
    assert accord_ou_refus("laisse tomber") is False
    # « Non, plutôt trois minutes » est une nouvelle demande, pas un refus sec.
    assert accord_ou_refus("non plutôt trois minutes") is None
    assert accord_ou_refus("") is None

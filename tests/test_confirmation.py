"""Actions irréversibles : Lily demande avant de faire."""

from __future__ import annotations

import asyncio
import json

from lily.core.apprentissage import Apprentissage
from lily.core.conversation import Conversation
from lily.core.engines.llm.mock import MockLLM
from lily.core.journal import JournalDesAppels
from lily.core.pipeline import Pipeline
from lily.core.registry import PluginRegistry
from lily.core.router import Router
from lily.core.text import accord_ou_refus

PLUGIN_DESTRUCTEUR = '''
from lily.plugin import skill

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


PLUGIN_IRREVERSIBLE = '''
from lily.plugin import skill

@skill(
    description="Supprime définitivement le contenu du bac.",
    examples=["vide le bac"],
    confirm="Voulez-vous vraiment que je vide le bac ?",
)
def tout_effacer() -> str:
    return "C'est vidé."
'''


def monter(dossier, llm=None):
    registry = PluginRegistry([dossier], data_root=dossier.parent / "data")
    registry.load_all()
    pipeline = Pipeline(
        registry, Router(llm or MockLLM()),
        Conversation(persona="persona de test", max_turns=4),
    )
    return pipeline, registry


def _pipeline(dossier, llm, magasin, journal=None):
    """Un pipeline branché sur un magasin d'apprentissage, pour observer ce
    qu'il retient — et à quel moment."""
    registry = PluginRegistry([dossier], data_root=dossier.parent / "data")
    registry.load_all()
    return Pipeline(
        registry,
        Router(llm, apprentissage=magasin, direct_threshold=0.99),
        Conversation(persona="persona de test", max_turns=4),
        apprentissage=magasin,
        journal=journal,
    )


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


def test_un_refus_ne_laisse_aucun_routage_appris(tmp_path, ecrire_plugin, dossier_plugins) -> None:
    """Le garde-fou annoncé : « un tour raté n'apprend rien ».

    Une compétence `confirm=` rend pourtant un résultat `ok=True` au moment où
    elle POSE sa question — et l'apprentissage se déclenchait là, avant toute
    réponse. Un « non » de correction laissait donc le mauvais routage en
    place, à défaire à la main.
    """
    ecrire_plugin("dangereux.py", PLUGIN_IRREVERSIBLE)
    magasin = Apprentissage(tmp_path / "m.sqlite")
    llm = MockLLM([json.dumps({"outil": "tout_effacer"}), "{}"])
    pipeline = _pipeline(dossier_plugins, llm, magasin)

    question = asyncio.run(pipeline.handle("débarrasse-moi de tout ce fouillis"))
    assert "vraiment" in question.speak.lower()
    assert magasin.statistiques()["phrases"] == 0, "rien n'est encore fait, rien à apprendre"

    refus = asyncio.run(pipeline.handle("non"))
    assert refus.speak == "Très bien, je n'ai rien fait."
    assert magasin.statistiques()["phrases"] == 0, "un refus ne doit rien laisser derrière lui"


def test_un_accord_apprend_la_phrase_d_origine(tmp_path, ecrire_plugin, dossier_plugins) -> None:
    """Et le pendant : c'est le « oui » qui transforme une intention en geste.

    La phrase retenue est celle qui a déclenché la question, pas le « oui » —
    c'est elle qui reviendra la prochaine fois.
    """
    ecrire_plugin("dangereux.py", PLUGIN_IRREVERSIBLE)
    magasin = Apprentissage(tmp_path / "m.sqlite")
    llm = MockLLM([json.dumps({"outil": "tout_effacer"}), "{}"])
    pipeline = _pipeline(dossier_plugins, llm, magasin)

    asyncio.run(pipeline.handle("débarrasse-moi de tout ce fouillis"))
    accord = asyncio.run(pipeline.handle("oui"))
    assert accord.skill_result.ok

    apprises = magasin.phrases_par_outil()
    assert list(apprises) == ["tout_effacer"]
    assert apprises["tout_effacer"][0].phrase == "débarrasse-moi de tout ce fouillis"


def test_un_geste_confirme_entre_au_journal(tmp_path, ecrire_plugin, dossier_plugins) -> None:
    """Le journal sert à « retiens cette routine » : un geste réellement
    exécuté doit s'y trouver, même s'il a fallu un accord pour l'obtenir."""
    ecrire_plugin("dangereux.py", PLUGIN_IRREVERSIBLE)
    journal = JournalDesAppels(5)
    llm = MockLLM([json.dumps({"outil": "tout_effacer"}), "{}"])
    pipeline = _pipeline(dossier_plugins, llm, Apprentissage(tmp_path / "m.sqlite"),
                         journal=journal)

    asyncio.run(pipeline.handle("débarrasse-moi de tout ce fouillis"))
    assert len(journal) == 0, "poser une question n'est pas un geste"
    asyncio.run(pipeline.handle("oui"))
    assert [appel.competence for appel in journal.recents()] == ["tout_effacer"]

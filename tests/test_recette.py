"""Le critère d'acceptation du projet, joué de bout en bout.

    1. J'écris un fichier plugins/dés.py contenant une fonction
       lancer_de(faces: int = 6) décorée @skill.
    2. Je le dépose dans le dossier pendant que Lily tourne.
    3. Je dis « Lily, lance un dé à vingt faces ».
    4. Elle répond avec un nombre entre 1 et 20.

Sans toucher au cœur, sans redémarrer, et — c'est le plus notable — sans
qu'aucun modèle de langage n'intervienne : l'étage déterministe du routeur
suffit.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from lily.app import build_assistant, start_hot_reload
from lily.core.config import Config
from lily.core.engines.llm.mock import MockLLM

# Exactement le fichier décrit par le critère d'acceptation, nom accentué compris.
DES = '''
import random

from lily.plugin import skill


@skill(
    description="Lance un dé et donne le résultat.",
    examples=["lance un dé", "lance un dé à six faces", "tire un dé"],
)
def lancer_de(faces: int = 6) -> str:
    """Lance un dé à N faces."""
    return str(random.randint(1, faces))
'''


def _assistant(dossier: Path, donnees: Path):
    config = Config({
        "profile": "pc",
        "assistant": {"persona_file": "config/persona.txt", "announce_new_skills": True},
        "plugins": {
            "paths": [str(dossier)],
            "data_dir": str(donnees),
            "hot_reload": True,
            "debounce_ms": 100,
        },
        "llm": {"engine": "mock", "router": {"direct_threshold": 0.72}},
    })
    return build_assistant(config, llm=MockLLM())


def test_le_critere_d_acceptation(dossier_plugins: Path, tmp_path: Path) -> None:
    pytest.importorskip("watchdog", reason="watchdog n'est pas installé")
    assistant = _assistant(dossier_plugins, tmp_path / "data")

    async def scenario() -> str:
        assistant.pipeline.attach()
        assert start_hot_reload(assistant), "la surveillance n'a pas démarré"
        assert assistant.registry.skills == {}

        # 1 et 2 : le fichier est déposé pendant que Lily tourne.
        (dossier_plugins / "dés.py").write_text(DES, encoding="utf-8")

        limite = time.monotonic() + 15
        while "lancer_de" not in assistant.registry.skills and time.monotonic() < limite:
            await asyncio.sleep(0.05)
        assert "lancer_de" in assistant.registry.skills, "le fichier déposé n'a pas été vu"

        # 3 et 4 : la phrase, et la réponse.
        resultat = await assistant.pipeline.handle("lance un dé à vingt faces")
        assert resultat.tier == "regle", "l'étage déterministe aurait dû suffire"
        assert assistant.llm.calls == [], "aucun modèle ne devait être sollicité"
        assert resultat.tool.arguments == {"faces": 20}
        return resultat.speak

    try:
        reponse = asyncio.run(scenario())
    finally:
        asyncio.run(assistant.aclose())

    assert 1 <= int(reponse) <= 20


def test_la_nouvelle_competence_est_annoncee(dossier_plugins: Path, tmp_path: Path) -> None:
    pytest.importorskip("watchdog", reason="watchdog n'est pas installé")
    assistant = _assistant(dossier_plugins, tmp_path / "data")
    dits: list[str] = []
    assistant.pipeline._speak = dits.append

    async def scenario() -> None:
        assistant.pipeline.attach()
        assert start_hot_reload(assistant)
        annonceur = asyncio.ensure_future(assistant.pipeline.run_announcer())

        (dossier_plugins / "dés.py").write_text(DES, encoding="utf-8")
        limite = time.monotonic() + 15
        while not dits and time.monotonic() < limite:
            await asyncio.sleep(0.05)
        annonceur.cancel()

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(assistant.aclose())

    assert dits == ["Nouvelle compétence disponible : lancer de."]


def test_le_fichier_retire_emporte_sa_competence(dossier_plugins: Path, tmp_path: Path) -> None:
    pytest.importorskip("watchdog", reason="watchdog n'est pas installé")
    assistant = _assistant(dossier_plugins, tmp_path / "data")
    chemin = dossier_plugins / "dés.py"

    async def scenario() -> None:
        assistant.pipeline.attach()
        assert start_hot_reload(assistant)

        chemin.write_text(DES, encoding="utf-8")
        limite = time.monotonic() + 15
        while "lancer_de" not in assistant.registry.skills and time.monotonic() < limite:
            await asyncio.sleep(0.05)
        assert "lancer_de" in assistant.registry.skills

        chemin.unlink()
        limite = time.monotonic() + 15
        while "lancer_de" in assistant.registry.skills and time.monotonic() < limite:
            await asyncio.sleep(0.05)
        assert "lancer_de" not in assistant.registry.skills

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(assistant.aclose())


def test_un_plugin_casse_depose_a_chaud_n_emporte_rien(
    dossier_plugins: Path, tmp_path: Path
) -> None:
    pytest.importorskip("watchdog", reason="watchdog n'est pas installé")
    assistant = _assistant(dossier_plugins, tmp_path / "data")

    async def scenario() -> None:
        assistant.pipeline.attach()
        assert start_hot_reload(assistant)

        (dossier_plugins / "dés.py").write_text(DES, encoding="utf-8")
        (dossier_plugins / "casse.py").write_text("raise RuntimeError('boum')\n", encoding="utf-8")

        limite = time.monotonic() + 15
        while (
            "lancer_de" not in assistant.registry.skills
            or not assistant.registry.failures()
        ) and time.monotonic() < limite:
            await asyncio.sleep(0.05)

        # Le fautif est écarté avec un message clair, l'autre fonctionne.
        assert [e.name for e in assistant.registry.failures()] == ["casse"]
        resultat = await assistant.pipeline.handle("lance un dé")
        assert 1 <= int(resultat.speak) <= 6

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(assistant.aclose())

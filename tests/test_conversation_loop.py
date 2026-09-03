"""La boucle complète : éveil, énoncé, réponse, suivi, barge-in.

Tout tourne sans micro, sans haut-parleur et sans modèle : le micro est un
tableau de trames en mémoire, l'éveil et le VAD sont scriptés.
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time

import pytest

from lily.core.audio import MemoryAudioInput, MemoryAudioOutput
from lily.core.conversation import Conversation
from lily.core.endpointer import BargeInDetector, Endpointer
from lily.core.engines.llm.mock import MockLLM
from lily.core.engines.stt.scripted import ScriptedSTT
from lily.core.engines.tts.silent import SilentTTS
from lily.core.engines.vad.scripted import ScriptedVAD
from lily.core.engines.wake.scripted import ScriptedWakeWord
from lily.core.listener import BargeInMode, ListenMode, VoiceListener
from lily.core.pipeline import Pipeline, State
from lily.core.registry import PluginRegistry
from lily.core.router import NO_TOOL, Router

TAILLE_MIC = 480

PLUGIN_HEURE = '''
from lily.plugin import skill

@skill(description="Donne l'heure.", examples=["quelle heure est-il"])
def heure() -> str:
    """Donne l'heure courante."""
    return "Il est midi."
'''

# Parle une dizaine de trames, puis se tait : de quoi terminer un énoncé.
PAROLE = [0.9] * 10 + [0.0] * 40


class SortieLente(MemoryAudioOutput):
    """Un haut-parleur qui prend le temps de jouer, et qu'on peut couper.

    Sans cela, la réponse entière est « jouée » en quelques microsecondes et
    aucun barge-in n'a le temps d'arriver — ce qui ne prouverait rien. Comme
    la vraie sortie, celle-ci consulte l'annulation pendant qu'elle joue.
    """

    def __init__(self, duree_s: float = 0.06) -> None:
        super().__init__()
        self.duree_s = duree_s

    def play(self, chunk, cancel: threading.Event | None = None) -> bool:
        echeance = time.monotonic() + self.duree_s
        while time.monotonic() < echeance:
            if cancel is not None and cancel.is_set():
                return False
            time.sleep(0.005)
        return super().play(chunk, cancel)


def trames(n: int = 40) -> list[bytes]:
    return [struct.pack(f"<{TAILLE_MIC}h", *([4000, -4000] * (TAILLE_MIC // 2)))] * n


async def attendre(predicat, timeout: float = 10.0) -> None:
    """Attend qu'une condition devienne vraie, sans faire tourner le CPU."""
    limite = asyncio.get_running_loop().time() + timeout
    while not predicat():
        if asyncio.get_running_loop().time() > limite:
            raise AssertionError("condition jamais atteinte")
        await asyncio.sleep(0.005)


def monter(
    dossier,
    *,
    dit: list[str],
    hits=(0,),
    probabilites=None,
    barge_in_probabilites=None,
    barge_in_mode=BargeInMode.VOICE,
    llm=None,
    follow_up_s: float = 5.0,
    on_state=None,
    sortie_lente: bool = False,
):
    registry = PluginRegistry([dossier], data_root=dossier.parent / "data")
    registry.load_all()
    sortie = SortieLente() if sortie_lente else MemoryAudioOutput()
    pipeline = Pipeline(
        registry,
        Router(llm or MockLLM()),
        Conversation(persona="persona de test", max_turns=4),
        stt=ScriptedSTT(dit),
        tts=SilentTTS(),
        audio_in=None,
        audio_out=sortie,
        echo=False,
        follow_up_s=follow_up_s,
        wake_beep=False,
        on_state=on_state,
    )
    vad = ScriptedVAD(probabilites or PAROLE, frame_size=512)
    listener = VoiceListener(
        MemoryAudioInput(trames(), 16000, TAILLE_MIC, repeat=True, frame_delay_s=0.001),
        endpointer=Endpointer(
            vad, min_speech_ms=64, silence_ms=200, pre_roll_ms=64,
            min_total_speech_ms=64, max_wait_s=100,
        ),
        on_event=pipeline.on_listener_event,
        wake=ScriptedWakeWord(hits, frame_size=1280),
        barge_in=BargeInDetector(
            ScriptedVAD(barge_in_probabilites or [0.0] * 500, frame_size=512),
            min_speech_ms=64, guard_ms=0,
        ),
        barge_in_mode=barge_in_mode,
        start_mode=ListenMode.PAUSED,
    )
    return pipeline, listener, sortie


async def faire_tourner(pipeline, listener, condition, *, use_wake=True, timeout=10.0):
    pipeline.attach()
    listener.start()
    arret = asyncio.Event()
    tache = asyncio.ensure_future(
        pipeline.run_conversation(listener, use_wake=use_wake, stop=arret)
    )
    try:
        await attendre(condition, timeout)
    finally:
        arret.set()
        listener.stop()
        with contextlib_suppress():
            await asyncio.wait_for(tache, timeout=5.0)
        pipeline.detach()


class contextlib_suppress:
    def __enter__(self): return self
    def __exit__(self, *exc): return True


def test_un_tour_complet_depuis_le_mot_d_eveil(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    pipeline, listener, sortie = monter(
        dossier_plugins, dit=["quelle heure est-il"], follow_up_s=0.0
    )

    asyncio.run(faire_tourner(pipeline, listener, lambda: bool(sortie.chunks)))

    assert sortie.texte == "Il est midi."


def test_le_mode_suivi_enchaine_sans_redire_le_nom(ecrire_plugin, dossier_plugins) -> None:
    # Le mot d'éveil ne se déclenche qu'une fois (hits=[0]) : si un second
    # tour a lieu, c'est bien que le mode suivi a gardé l'écoute ouverte.
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    pipeline, listener, sortie = monter(
        dossier_plugins, dit=["quelle heure est-il", "quelle heure est-il"], follow_up_s=5.0
    )

    asyncio.run(faire_tourner(pipeline, listener, lambda: len(sortie.chunks) >= 2))

    assert [c.text for c in sortie.chunks[:2]] == ["Il est midi.", "Il est midi."]


def test_sans_mode_suivi_on_retourne_attendre_le_mot_d_eveil(
    ecrire_plugin, dossier_plugins
) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    pipeline, listener, sortie = monter(
        dossier_plugins, dit=["quelle heure est-il", "et maintenant"], follow_up_s=0.0
    )

    async def scenario() -> None:
        await faire_tourner(pipeline, listener, lambda: bool(sortie.chunks))
        # Après la réponse, l'écoute est revenue en attente du mot d'éveil.
        assert listener.mode is ListenMode.WAKE or listener.pending is ListenMode.WAKE

    asyncio.run(scenario())
    assert len(sortie.chunks) == 1


def test_sans_mot_d_eveil_lily_ecoute_tout(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    pipeline, listener, sortie = monter(
        dossier_plugins, dit=["quelle heure est-il"], hits=(), follow_up_s=0.0
    )

    asyncio.run(faire_tourner(
        pipeline, listener, lambda: bool(sortie.chunks), use_wake=False
    ))
    assert sortie.texte == "Il est midi."


def test_une_transcription_vide_ne_declenche_aucun_tour(
    ecrire_plugin, dossier_plugins
) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    pipeline, listener, sortie = monter(dossier_plugins, dit=[""], follow_up_s=0.0)

    async def scenario() -> None:
        await faire_tourner(
            pipeline, listener,
            lambda: listener.mode is ListenMode.WAKE or listener.pending is ListenMode.WAKE,
        )

    asyncio.run(scenario())
    assert sortie.chunks == []
    assert len(pipeline.conversation) == 0


def test_on_peut_couper_la_parole_a_lily(ecrire_plugin, dossier_plugins) -> None:
    # Réponse longue en plusieurs phrases ; l'utilisateur reprend la parole
    # après la première. Lily doit se taire au milieu et rouvrir l'écoute.
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    llm = MockLLM([json.dumps({"outil": NO_TOOL}), "Une. Deux. Trois. Quatre. Cinq."])
    pipeline, listener, sortie = monter(
        dossier_plugins,
        dit=["raconte-moi quelque chose de long"],
        llm=llm,
        # Silence au début de la réponse, puis parole soutenue.
        barge_in_probabilites=[0.0] * 60 + [1.0] * 500,
        follow_up_s=0.0,
        sortie_lente=True,
    )

    asyncio.run(faire_tourner(
        pipeline, listener,
        lambda: bool(sortie.chunks) and pipeline._barge_in.is_set(),
    ))

    # Elle a commencé à parler, mais pas tout dit.
    assert 0 < len(sortie.chunks) < 5


def test_le_barge_in_coupe_pendant_la_parole_pas_apres(
    ecrire_plugin, dossier_plugins
) -> None:
    # Piège trouvé en écrivant ce test : la boucle de conversation attend la
    # fin de `handle_and_speak`, donc un événement mis en file n'était traité
    # qu'une fois la réponse entièrement prononcée — trop tard pour
    # l'interrompre. Le drapeau est désormais armé depuis le thread d'écoute.
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    llm = MockLLM([json.dumps({"outil": NO_TOOL}), "Une. Deux. Trois. Quatre. Cinq."])
    pipeline, listener, sortie = monter(
        dossier_plugins, dit=["parle"], llm=llm,
        barge_in_probabilites=[1.0] * 500, follow_up_s=0.0, sortie_lente=True,
    )

    asyncio.run(faire_tourner(
        pipeline, listener, lambda: pipeline._barge_in.is_set()
    ))
    # Coupée avant même la première phrase : l'utilisateur parlait déjà.
    assert len(sortie.chunks) < 5


def test_le_barge_in_desactive_laisse_finir_la_phrase(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    llm = MockLLM([json.dumps({"outil": NO_TOOL}), "Une. Deux. Trois."])
    pipeline, listener, sortie = monter(
        dossier_plugins,
        dit=["raconte"],
        llm=llm,
        barge_in_probabilites=[1.0] * 500,
        barge_in_mode=BargeInMode.OFF,
        follow_up_s=0.0,
    )

    asyncio.run(faire_tourner(pipeline, listener, lambda: len(sortie.chunks) >= 3))
    assert [c.text for c in sortie.chunks] == ["Une.", "Deux.", "Trois."]


def test_les_etats_du_tour_vocal_complet(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    etats: list[State] = []
    pipeline, listener, sortie = monter(
        dossier_plugins, dit=["quelle heure est-il"],
        follow_up_s=0.0, on_state=etats.append,
    )

    asyncio.run(faire_tourner(pipeline, listener, lambda: bool(sortie.chunks)))

    # La machine à états passe bien par l'éveil, ce qui n'existait pas à
    # l'étape 2, puis retombe au repos.
    assert etats[:4] == [State.WAKE, State.LISTEN, State.TRANSCRIBE, State.THINK]  # sans repos parasite
    assert State.ACT in etats and State.SPEAK in etats
    assert etats[-1] is State.IDLE


@pytest.mark.parametrize("mode", [BargeInMode.VOICE, BargeInMode.WAKE, BargeInMode.OFF])
def test_les_trois_modes_de_barge_in_sont_acceptes(mode) -> None:
    assert BargeInMode(str(mode)) is mode


# --- montage réel depuis une configuration ---------------------------------

def test_le_montage_de_l_ecoute_suit_la_configuration(tmp_path, dossier_plugins) -> None:
    # Ce test existe parce qu'un bloc d'imports oublié dans app.py n'aurait
    # été découvert qu'au premier démarrage en mode vocal : aucun test
    # n'appelait build_listener.
    from lily.app import build_assistant, build_listener
    from lily.core.config import Config
    from lily.core.engines.vad.scripted import ScriptedVAD

    config = Config({
        "profile": "pc",
        "assistant": {"persona_file": "config/persona.txt", "follow_up_seconds": 3.0},
        "plugins": {"paths": [str(dossier_plugins)], "data_dir": str(tmp_path / "data")},
        "llm": {"engine": "mock"},
        "audio": {"sample_rate": 16000, "frame_ms": 30},
        "vad": {"engine": "scripted", "threshold": 0.4, "silence_ms": 900, "pre_roll_ms": 250},
        "barge_in": {"mode": "eveil", "threshold": 0.9, "min_speech_ms": 250},
    })
    assistant = build_assistant(
        config, llm=MockLLM(), voice=True,
        stt=ScriptedSTT([]), tts=SilentTTS(),
        audio_in=MemoryAudioInput(trames(5), 16000, TAILLE_MIC),
        audio_out=MemoryAudioOutput(),
    )
    try:
        listener = build_listener(
            assistant,
            vad=ScriptedVAD([], frame_size=512),
            barge_in_vad=ScriptedVAD([], frame_size=512),
            wake=ScriptedWakeWord(),
        )
        assert listener.endpointer.threshold == pytest.approx(0.4)
        assert listener.endpointer.silence_ms == pytest.approx(900)
        assert listener.endpointer.pre_roll_ms == pytest.approx(250)
        assert listener.barge_in_mode is BargeInMode.WAKE
        assert listener.barge_in.threshold == pytest.approx(0.9)
        # Deux instances de VAD distinctes : partager l'état interne du modèle
        # entre l'écoute et la surveillance les ferait interférer.
        assert listener.barge_in.vad is not listener.endpointer.vad
        assert assistant.pipeline.follow_up_s == pytest.approx(3.0)
    finally:
        asyncio.run(assistant.aclose())

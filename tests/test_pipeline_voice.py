"""Chaîne vocale complète, sans micro, sans haut-parleur, sans modèle.

C'est le pendant de la promesse de l'étape 1 : la boucle a été prouvée sans
audio, et l'audio est prouvé sans matériel.
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading

from capucine.core.audio import AudioChunk, MemoryAudioInput, MemoryAudioOutput
from capucine.core.conversation import Conversation
from capucine.core.engines.llm.mock import MockLLM
from capucine.core.engines.stt.scripted import ScriptedSTT
from capucine.core.engines.tts.silent import SilentTTS
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


def trames(secondes: float = 1.0, sample_rate: int = 16000) -> list[bytes]:
    """Des trames de 30 ms d'un signal audible."""
    par_trame = int(sample_rate * 0.03)
    n = int(secondes / 0.03)
    return [struct.pack(f"<{par_trame}h", *([6000, -6000] * (par_trame // 2)))] * n


_DEFAUT = object()


def monter(dossier, *, llm=None, dit=None, tts=_DEFAUT, sortie=None, entree=None, **kwargs):
    llm = llm or MockLLM()
    registry = PluginRegistry([dossier], data_root=dossier.parent / "data")
    registry.load_all()
    sortie = sortie if sortie is not None else MemoryAudioOutput()
    pipeline = Pipeline(
        registry,
        Router(llm),
        Conversation(persona="persona de test", max_turns=3),
        stt=ScriptedSTT(dit or []),
        tts=SilentTTS() if tts is _DEFAUT else tts,
        audio_in=entree if entree is not None else MemoryAudioInput(trames(), 16000, 480),
        audio_out=sortie,
        echo=False,
        **kwargs,
    )
    return pipeline, sortie, llm


def test_un_tour_vocal_complet(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    pipeline, sortie, llm = monter(dossier_plugins, dit=["quelle heure est-il"])

    resultat = asyncio.run(pipeline.voice_turn())

    assert resultat.utterance == "quelle heure est-il"
    assert resultat.tier == "regle"
    assert resultat.speak == "Il est midi."
    assert sortie.texte == "Il est midi."       # la parole est bien passée par le TTS
    assert llm.calls == []                      # aucun modèle sollicité
    # Les latences des étages audio rejoignent celles du tour.
    assert "transcription_ms" in resultat.telemetry.stages
    assert "audio_s" in resultat.telemetry.stages


def test_les_etats_du_chemin_vocal(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    etats: list[State] = []
    pipeline, _, _ = monter(dossier_plugins, dit=["quelle heure est-il"], on_state=etats.append)

    asyncio.run(pipeline.voice_turn())

    assert etats == [
        State.LISTEN, State.IDLE, State.TRANSCRIBE, State.IDLE,
        State.THINK, State.ACT, State.SPEAK, State.IDLE,
    ]


def test_un_silence_ne_declenche_aucun_tour(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    pipeline, sortie, llm = monter(dossier_plugins, dit=[""])  # transcription vide

    resultat = asyncio.run(pipeline.voice_turn())

    assert resultat.utterance == ""
    assert sortie.chunks == []
    assert llm.calls == []
    assert len(pipeline.conversation) == 0


def test_la_reponse_conversationnelle_est_dite_phrase_par_phrase(
    ecrire_plugin, dossier_plugins
) -> None:
    # C'est le gain de latence : la première phrase part au haut-parleur
    # pendant que le modèle écrit la seconde.
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    llm = MockLLM([json.dumps({"outil": NO_TOOL}), "Il est midi. Le soleil brille."])
    pipeline, sortie, _ = monter(dossier_plugins, llm=llm, dit=["raconte-moi ta journée"])

    resultat = asyncio.run(pipeline.voice_turn())

    assert resultat.tier == "conversation"
    assert [c.text for c in sortie.chunks] == ["Il est midi.", "Le soleil brille."]
    assert resultat.speak == "Il est midi. Le soleil brille."
    assert pipeline.conversation.history()[-1].content == "Il est midi. Le soleil brille."


def test_la_parole_s_interrompt_en_cours_de_route(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)

    class SortieQuiCoupe(MemoryAudioOutput):
        """Simule l'utilisateur qui reprend la parole après la 1re phrase."""

        def __init__(self, drapeau: threading.Event) -> None:
            super().__init__()
            self.drapeau = drapeau

        def play(self, chunk: AudioChunk, cancel: threading.Event | None = None) -> bool:
            joue = super().play(chunk, cancel)
            self.drapeau.set()
            return joue

    llm = MockLLM([json.dumps({"outil": NO_TOOL}), "Une. Deux. Trois."])
    drapeau = threading.Event()
    sortie = SortieQuiCoupe(drapeau)
    pipeline, _, _ = monter(dossier_plugins, llm=llm, dit=["parle-moi"], sortie=sortie)
    drapeau = pipeline._barge_in
    sortie.drapeau = drapeau

    resultat = asyncio.run(pipeline.voice_turn())

    assert resultat.interrupted
    assert [c.text for c in sortie.chunks] == ["Une."]


def test_couper_un_tour_arme_le_drapeau_et_stoppe_la_sortie(dossier_plugins) -> None:
    class SortieTracee(MemoryAudioOutput):
        arrets = 0

        def stop(self) -> None:
            SortieTracee.arrets += 1

    pipeline, _, _ = monter(dossier_plugins, sortie=SortieTracee())
    pipeline.cancel_turn()
    assert pipeline._barge_in.is_set()
    assert SortieTracee.arrets == 1


def test_une_synthese_en_panne_n_empeche_pas_de_repondre(ecrire_plugin, dossier_plugins) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)

    class TTSQuiCasse(SilentTTS):
        def synthesize(self, text, cancel=None):
            raise OSError("carte son disparue")
            yield  # pragma: no cover

    pipeline, sortie, _ = monter(
        dossier_plugins, dit=["quelle heure est-il"], tts=TTSQuiCasse()
    )
    resultat = asyncio.run(pipeline.voice_turn())

    assert resultat.speak == "Il est midi."   # la réponse existe malgré tout
    assert sortie.chunks == []


def test_sans_voix_capucine_affiche_au_lieu_de_dire(ecrire_plugin, dossier_plugins, capsys) -> None:
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    pipeline, _, _ = monter(dossier_plugins, dit=["quelle heure est-il"], tts=None)
    assert not pipeline.has_voice

    asyncio.run(pipeline.voice_turn())
    assert "Il est midi." in capsys.readouterr().out


def test_la_capture_s_arrete_sur_le_signal_du_vad(ecrire_plugin, dossier_plugins) -> None:
    # L'étape 3 armera ce signal depuis le VAD ; le pipeline est déjà prêt.
    ecrire_plugin("horloge.py", PLUGIN_HEURE)
    entree = MemoryAudioInput(trames(secondes=30), 16000, 480)
    pipeline, _, _ = monter(dossier_plugins, dit=["quelle heure est-il"], entree=entree)

    stop = threading.Event()
    stop.set()
    audio = asyncio.run(pipeline.listen(stop=stop))
    assert audio.duration_s < 0.1


def test_la_duree_maximale_borne_la_capture(ecrire_plugin, dossier_plugins) -> None:
    entree = MemoryAudioInput(trames(secondes=60), 16000, 480)
    pipeline, _, _ = monter(dossier_plugins, entree=entree, max_utterance_s=1.0)
    audio = asyncio.run(pipeline.listen())
    assert audio.duration_s <= 1.05


def test_un_peripherique_qui_disparait_bascule_en_affichage(
    ecrire_plugin, dossier_plugins, capsys
) -> None:
    from capucine.core.audio import AudioUnavailable

    ecrire_plugin("horloge.py", PLUGIN_HEURE)

    class SortieQuiDisparait(MemoryAudioOutput):
        def play(self, chunk, cancel=None):
            raise AudioUnavailable("carte son débranchée")

    pipeline, _, _ = monter(
        dossier_plugins, dit=["quelle heure est-il"], sortie=SortieQuiDisparait()
    )
    pipeline.echo = False
    resultat = asyncio.run(pipeline.voice_turn())

    assert resultat.speak == "Il est midi."
    assert "Il est midi." in capsys.readouterr().out
    # La sortie est abandonnée pour le reste de la session : on ne répète pas
    # l'erreur à chaque phrase.
    assert pipeline.audio_out is None
    assert not pipeline.has_voice

"""Le fil qui tient le micro : un mode, un traitement, un événement."""

from __future__ import annotations

import struct

from capucine.core.audio import MemoryAudioInput
from capucine.core.endpointer import BargeInDetector, Endpointer
from capucine.core.engines.vad.scripted import ScriptedVAD
from capucine.core.engines.wake.scripted import ScriptedWakeWord
from capucine.core.listener import BargeInMode, ListenerEvent, ListenMode, VoiceListener

TAILLE_MIC = 480     # 30 ms à 16 kHz, ce que délivre la configuration par défaut


def trames(n: int, amplitude: int = 4000) -> list[bytes]:
    return [struct.pack(f"<{TAILLE_MIC}h", *([amplitude, -amplitude] * (TAILLE_MIC // 2)))] * n


def derouler(listener: VoiceListener, evenements: list[ListenerEvent]) -> list[str]:
    """Fait tourner l'écoute jusqu'à épuisement des trames."""
    listener.start()
    listener.stop(timeout=5.0)
    return [e.kind for e in evenements]


def monter(mode, *, probabilites=None, hits=(), barge_in_mode=BargeInMode.VOICE, n_trames=60):
    evenements: list[ListenerEvent] = []
    vad = ScriptedVAD(probabilites or [], frame_size=512)
    listener = VoiceListener(
        MemoryAudioInput(trames(n_trames), 16000, TAILLE_MIC),
        endpointer=Endpointer(vad, min_speech_ms=64, silence_ms=200,
                              pre_roll_ms=64, min_total_speech_ms=64, max_wait_s=100),
        on_event=evenements.append,
        wake=ScriptedWakeWord(hits, frame_size=1280),
        barge_in=BargeInDetector(ScriptedVAD(probabilites or [], frame_size=512),
                                 min_speech_ms=64, guard_ms=0),
        barge_in_mode=barge_in_mode,
        start_mode=mode,
    )
    return listener, evenements


def test_en_pause_rien_ne_sort() -> None:
    listener, evenements = monter(ListenMode.PAUSED, probabilites=[1.0] * 100, hits=[0])
    assert derouler(listener, evenements) == ["stopped"]


def test_en_mode_eveil_le_mot_declenche_puis_l_ecoute_se_met_en_pause() -> None:
    listener, evenements = monter(ListenMode.WAKE, hits=[0])
    kinds = derouler(listener, evenements)
    # Une seule détection : sans la mise en pause automatique, la même
    # détection ressortirait à chaque trame suivante.
    assert kinds == ["wake", "stopped"]
    assert listener.mode is ListenMode.PAUSED


def test_les_trames_sont_regroupees_pour_le_moteur_d_eveil() -> None:
    # Le micro délivre 480 échantillons, openWakeWord en veut 1280.
    listener, evenements = monter(ListenMode.WAKE, hits=[1], n_trames=30)
    assert derouler(listener, evenements) == ["wake", "stopped"]


def test_en_mode_enonce_la_fin_de_phrase_est_emise() -> None:
    listener, evenements = monter(
        ListenMode.UTTERANCE, probabilites=[0.9] * 10 + [0.0] * 40
    )
    kinds = derouler(listener, evenements)
    assert kinds == ["utterance", "stopped"]
    enonce = evenements[0].utterance
    assert enonce is not None and enonce
    assert listener.mode is ListenMode.PAUSED


def test_un_silence_prolonge_emet_un_enonce_vide() -> None:
    listener, evenements = monter(ListenMode.UTTERANCE, probabilites=[0.0] * 200)
    listener.endpointer.max_wait_s = 0.2
    kinds = derouler(listener, evenements)
    assert kinds == ["utterance", "stopped"]
    # Fini, mais rien à transcrire : le consommateur doit tester la valeur de
    # vérité, pas seulement la présence de l'événement.
    assert not evenements[0].utterance


def test_pendant_la_reponse_une_parole_soutenue_interrompt() -> None:
    listener, evenements = monter(ListenMode.MONITOR, probabilites=[1.0] * 100)
    assert derouler(listener, evenements) == ["barge_in", "stopped"]


def test_le_barge_in_peut_etre_desactive() -> None:
    listener, evenements = monter(
        ListenMode.MONITOR, probabilites=[1.0] * 100, barge_in_mode=BargeInMode.OFF
    )
    assert derouler(listener, evenements) == ["stopped"]


def test_en_mode_eveil_seul_le_mot_interrompt() -> None:
    # Sur haut-parleur, le micro entend la réponse en cours : n'accepter que
    # « Capucine » évite qu'elle se coupe elle-même.
    listener, evenements = monter(
        ListenMode.MONITOR, probabilites=[1.0] * 100, hits=[], barge_in_mode=BargeInMode.WAKE
    )
    assert derouler(listener, evenements) == ["stopped"]

    listener, evenements = monter(
        ListenMode.MONITOR, probabilites=[1.0] * 100, hits=[0], barge_in_mode=BargeInMode.WAKE
    )
    assert derouler(listener, evenements) == ["barge_in", "stopped"]


def test_un_moteur_qui_leve_ne_tue_pas_l_ecoute() -> None:
    class EveilQuiCasse(ScriptedWakeWord):
        def process(self, frame):
            raise RuntimeError("modèle corrompu")

    evenements: list[ListenerEvent] = []
    listener = VoiceListener(
        MemoryAudioInput(trames(30), 16000, TAILLE_MIC),
        endpointer=Endpointer(ScriptedVAD([], frame_size=512)),
        on_event=evenements.append,
        wake=EveilQuiCasse(frame_size=1280),
        start_mode=ListenMode.WAKE,
    )
    # Le thread va jusqu'au bout et signale son arrêt normalement.
    assert derouler(listener, evenements) == ["stopped"]
    assert listener.mode is ListenMode.PAUSED


def test_un_consommateur_qui_leve_ne_tue_pas_l_ecoute() -> None:
    appels: list[str] = []

    def consommateur(evenement: ListenerEvent) -> None:
        appels.append(evenement.kind)
        raise RuntimeError("consommateur fâché")

    listener = VoiceListener(
        MemoryAudioInput(trames(30), 16000, TAILLE_MIC),
        endpointer=Endpointer(ScriptedVAD([], frame_size=512)),
        on_event=consommateur,
        wake=ScriptedWakeWord([0], frame_size=1280),
        start_mode=ListenMode.WAKE,
    )
    listener.start()
    listener.stop(timeout=5.0)
    assert appels == ["wake", "stopped"]

"""Transport audio : tout doit fonctionner sans micro ni haut-parleur."""

from __future__ import annotations

import struct
import threading
import wave
from pathlib import Path

import pytest

from lily.core.audio import (
    AudioBuffer,
    AudioChunk,
    AudioUnavailable,
    MemoryAudioInput,
    MemoryAudioOutput,
    NullAudioOutput,
    SoundDeviceInput,
    WavFileInput,
    WavFileOutput,
    list_devices,
    record,
)


def pcm(valeurs: list[int]) -> bytes:
    return struct.pack(f"<{len(valeurs)}h", *valeurs)


def test_un_tampon_connait_sa_duree_et_son_niveau() -> None:
    buffer = AudioBuffer(pcm([16384, -16384] * 8000), 16000)
    assert buffer.n_samples == 16000
    assert buffer.duration_s == pytest.approx(1.0)
    assert buffer.rms() == pytest.approx(0.5, abs=0.01)
    assert AudioBuffer(b"", 16000).rms() == 0.0
    assert not AudioBuffer(b"", 16000)


def test_la_conversion_en_flottants_reste_dans_les_bornes() -> None:
    np = pytest.importorskip("numpy")
    tableau = AudioBuffer(pcm([32767, -32768, 0]), 16000).to_float32()
    assert tableau.dtype == np.float32
    assert tableau.max() <= 1.0 and tableau.min() >= -1.0


def test_capture_en_memoire_sans_le_moindre_peripherique() -> None:
    trames = [pcm([100] * 480) for _ in range(5)]
    source = MemoryAudioInput(trames, 16000, 480)
    source.start()
    buffer = record(source, max_seconds=10)
    assert buffer.n_samples == 5 * 480
    assert buffer.sample_rate == 16000


def test_la_capture_s_arrete_sur_le_signal() -> None:
    # C'est ainsi que le VAD de l'étape 3 terminera un énoncé.
    stop = threading.Event()
    stop.set()
    source = MemoryAudioInput([pcm([1] * 480)] * 100, 16000, 480)
    source.start()
    assert record(source, stop=stop).n_samples == 480  # la trame en cours, puis on sort


def test_la_capture_est_bornee_en_duree() -> None:
    source = MemoryAudioInput([pcm([1] * 1600)] * 100, 16000, 1600)
    source.start()
    buffer = record(source, max_seconds=0.5)
    assert buffer.duration_s == pytest.approx(0.5, abs=0.1)


def test_aller_retour_par_fichier_wav(tmp_path: Path) -> None:
    chemin = tmp_path / "entree.wav"
    with wave.open(str(chemin), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm([500] * 4800))

    source = WavFileInput(chemin, frame_ms=30)
    source.start()
    buffer = record(source)
    assert buffer.sample_rate == 16000
    assert buffer.n_samples == 4800

    sortie = tmp_path / "sortie.wav"
    destination = WavFileOutput(sortie)
    destination.play(AudioChunk(pcm([1000] * 2205), 22050, "bonjour"))
    with wave.open(str(sortie)) as handle:
        assert handle.getframerate() == 22050
        assert handle.getnframes() == 2205


def test_un_wav_stereo_est_refuse_avec_un_message_clair(tmp_path: Path) -> None:
    chemin = tmp_path / "stereo.wav"
    with wave.open(str(chemin), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm([0] * 100))
    with pytest.raises(AudioUnavailable, match="mono"):
        WavFileInput(chemin)


def test_la_sortie_en_memoire_respecte_l_interruption() -> None:
    sortie = MemoryAudioOutput()
    cancel = threading.Event()
    assert sortie.play(AudioChunk(pcm([1] * 10), 22050, "première"), cancel=cancel)
    cancel.set()
    assert not sortie.play(AudioChunk(pcm([1] * 10), 22050, "seconde"), cancel=cancel)
    assert sortie.texte == "première"
    assert [c.text for c in sortie.interrompus] == ["seconde"]


def test_la_sortie_nulle_ne_joue_rien_mais_respecte_l_interruption() -> None:
    sortie = NullAudioOutput()
    assert sortie.play(AudioChunk(b"\x00" * 100, 22050))
    cancel = threading.Event()
    cancel.set()
    assert not sortie.play(AudioChunk(b"\x00" * 100, 22050), cancel=cancel)


def test_l_inventaire_des_peripheriques_ne_leve_jamais() -> None:
    # Avec ou sans PortAudio, on rend une chaîne exploitable.
    assert isinstance(list_devices(), str)


def test_le_micro_reel_annonce_ce_qu_il_faut_installer() -> None:
    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError):
        with pytest.raises(AudioUnavailable, match="pip install sounddevice"):
            SoundDeviceInput().start()
    else:  # pragma: no cover - dépend de la machine
        pytest.skip("PortAudio est installé sur cette machine")


def test_un_morceau_connait_sa_duree() -> None:
    assert AudioChunk(b"\x00" * 44100, 22050).duration_s == pytest.approx(1.0)


# --- montage des périphériques ---------------------------------------------

def test_sans_micro_le_montage_echoue_tout_de_suite(monkeypatch) -> None:
    # Mieux vaut refuser au démarrage avec un message net qu'échouer à chaque
    # phrase prononcée.
    from lily.core.engines import factory
    from lily.core.errors import EngineUnavailable

    monkeypatch.setattr("lily.core.audio._peripherique_disponible", lambda *_: False)
    config = _config_audio()
    with pytest.raises(EngineUnavailable, match="Aucun micro"):
        factory.build_audio_input(config)


def test_sans_haut_parleur_lily_affiche_au_lieu_de_dire(monkeypatch) -> None:
    from lily.core.engines import factory

    monkeypatch.setattr("lily.core.audio._peripherique_disponible", lambda *_: False)
    assert factory.build_audio_output(_config_audio()) is None


def test_avec_un_peripherique_on_obtient_bien_les_flux_reels(monkeypatch) -> None:
    from lily.core.audio import SoundDeviceOutput
    from lily.core.engines import factory

    monkeypatch.setattr("lily.core.audio._peripherique_disponible", lambda *_: True)
    assert isinstance(factory.build_audio_output(_config_audio()), SoundDeviceOutput)


def test_le_mode_muet_garde_la_synthese_mais_ne_joue_rien() -> None:
    from lily.core.engines import factory

    assert isinstance(factory.build_audio_output(_config_audio(), silent=True), NullAudioOutput)


def _config_audio():
    from lily.core.config import Config

    return Config({"audio": {"sample_rate": 16000, "frame_ms": 30}})


# --- arrêt du micro et écriture du WAV --------------------------------------

def test_le_micro_rend_la_main_meme_la_file_pleine() -> None:
    """`stop()` déposait une sentinelle « si la file n'est pas pleine ». Or elle
    l'est justement quand la machine peine — c'est-à-dire quand on veut
    s'arrêter — et `frames()` restait alors bloqué pour toujours."""
    entree = SoundDeviceInput(queue_frames=2)
    for _ in range(2):
        entree._queue.put_nowait(b"\x00" * 960)
    assert entree._queue.full()

    lues: list[bytes] = []
    fini = threading.Event()

    def consommer() -> None:
        for frame in entree.frames():
            lues.append(frame)
        fini.set()

    fil = threading.Thread(target=consommer, daemon=True)
    fil.start()
    entree.stop()
    assert fini.wait(3.0), "frames() n'a jamais rendu la main"
    fil.join(1.0)
    assert lues, "les trames déjà captées doivent être rendues avant l'arrêt"


def test_le_wav_n_est_ecrit_qu_une_fois_par_phrase(tmp_path: Path) -> None:
    """Chaque phrase réécrivait le fichier entier depuis le début : un coût
    quadratique. On vérifie le contenu, et qu'aucune phrase n'est réécrite."""
    cible = tmp_path / "sortie.wav"
    sortie = WavFileOutput(cible)
    ecrits: list[int] = []
    for i in range(5):
        assert sortie.play(AudioChunk(pcm=bytes([i]) * 320, sample_rate=16000))
        ecrits.append(cible.stat().st_size)
    sortie.close()

    # Le fichier ne fait que croître d'une phrase, jamais de tout l'historique.
    assert ecrits == [44 + 320 * (i + 1) for i in range(5)]
    with wave.open(str(cible), "rb") as handle:
        assert handle.getframerate() == 16000
        assert handle.readframes(handle.getnframes()) == b"".join(
            bytes([i]) * 320 for i in range(5)
        )


def test_le_wav_reste_lisible_avant_sa_fermeture(tmp_path: Path) -> None:
    """Garder le descripteur ouvert ne doit pas rendre le fichier illisible en
    cours de séance : l'en-tête est remis à jour à chaque phrase."""
    cible = tmp_path / "en_cours.wav"
    sortie = WavFileOutput(cible)
    sortie.play(AudioChunk(pcm=b"\x01\x02" * 160, sample_rate=22050))
    with wave.open(str(cible), "rb") as handle:   # sans close() préalable
        assert handle.getframerate() == 22050
        assert handle.getnframes() == 160
    sortie.close()

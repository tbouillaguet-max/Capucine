"""Détection d'activité vocale."""

from __future__ import annotations

import struct

import pytest

from capucine.core.audio import Rechunker
from capucine.core.engines.vad.energy import EnergyVAD, rms
from capucine.core.engines.vad.scripted import ScriptedVAD
from capucine.core.engines.vad.silero import SileroVAD
from capucine.core.errors import EngineUnavailable


def trame(amplitude: int, taille: int = 480) -> bytes:
    return struct.pack(f"<{taille}h", *([amplitude, -amplitude] * (taille // 2)))


# --- rechunker -------------------------------------------------------------

def test_le_rechunker_rend_des_trames_de_taille_exacte() -> None:
    # Silero exige 512 échantillons, openWakeWord 1280, le micro délivre 480 :
    # sans regroupement, rien ne s'emboîte.
    rechunker = Rechunker(512)
    trames = list(rechunker.push(b"\x00" * (480 * 2)))
    assert trames == []                      # pas encore assez
    assert rechunker.en_attente == 960

    trames = list(rechunker.push(b"\x00" * (480 * 2)))
    assert [len(t) for t in trames] == [1024]
    assert rechunker.en_attente == 1920 - 1024

    rechunker.reset()
    assert rechunker.en_attente == 0


def test_une_taille_de_trame_nulle_est_refusee() -> None:
    with pytest.raises(ValueError):
        Rechunker(0)


# --- VAD par énergie -------------------------------------------------------

def test_le_vad_par_energie_distingue_le_silence_de_la_voix() -> None:
    vad = EnergyVAD()
    assert vad.available()
    assert vad.speech_probability(trame(0)) == 0.0
    assert vad.speech_probability(trame(9000)) == 1.0


def test_le_plancher_de_bruit_absorbe_un_souffle_continu() -> None:
    # Un ventilateur qui se met en marche ne doit pas passer indéfiniment pour
    # de la parole : le plancher monte, lentement, même au-dessus du seuil.
    vad = EnergyVAD(noise_floor=0.0001, creep=0.05, margin=3.0)
    souffle = trame(600)
    premier = vad.speech_probability(souffle)
    for _ in range(300):
        vad.speech_probability(souffle)
    assert premier == 1.0
    assert vad.speech_probability(souffle) == 0.0


def test_une_phrase_normale_ne_fait_pas_monter_le_plancher() -> None:
    # Le glissement doit être assez lent pour qu'une phrase de quelques
    # secondes reste détectée du début à la fin.
    vad = EnergyVAD()
    voix = trame(9000)
    scores = [vad.speech_probability(voix) for _ in range(int(5 / 0.03))]
    assert all(score == 1.0 for score in scores)


def test_le_calcul_de_niveau_est_robuste() -> None:
    assert rms(b"") == 0.0
    assert rms(b"\x00") == 0.0            # octet impair, pas de plantage
    assert rms(trame(16384)) == pytest.approx(0.5, abs=0.01)


# --- VAD scripté -----------------------------------------------------------

def test_le_vad_scripte_suit_sa_liste() -> None:
    vad = ScriptedVAD([0.1, 0.9], default=0.4)
    assert [vad.speech_probability(b"") for _ in range(3)] == [0.1, 0.9, 0.4]
    vad.reset()
    assert vad.speech_probability(b"") == 0.1


# --- Silero ----------------------------------------------------------------

def test_silero_trouve_son_modele_sans_importer_torch() -> None:
    # Le paquet silero-vad importe torch dès son __init__, y compris sur le
    # chemin ONNX. On localise donc le fichier sans importer le paquet.
    vad = SileroVAD()
    if not vad.available():
        pytest.skip("silero-vad n'est pas installé")
    assert vad.model_path() is not None
    assert vad.model_path().name.endswith(".onnx")
    assert vad.frame_size == 512
    assert "torch" not in __import__("sys").modules


def test_silero_rend_une_probabilite_sur_du_vrai_audio() -> None:
    vad = SileroVAD()
    if not vad.available():
        pytest.skip("silero-vad n'est pas installé")
    probabilite = vad.speech_probability(b"\x00\x00" * 512)
    assert 0.0 <= probabilite <= 1.0
    assert probabilite < 0.5        # du silence n'est pas de la parole
    vad.reset()


def test_silero_refuse_une_trame_de_mauvaise_taille() -> None:
    vad = SileroVAD()
    if not vad.available():
        pytest.skip("silero-vad n'est pas installé")
    with pytest.raises(EngineUnavailable, match="Rechunker"):
        vad.speech_probability(b"\x00\x00" * 480)


def test_silero_dit_comment_obtenir_son_modele(tmp_path) -> None:
    vad = SileroVAD(model_path=tmp_path / "absent.onnx", models_dir=tmp_path)
    assert not vad.available()
    with pytest.raises(EngineUnavailable, match="no-deps silero-vad"):
        vad.speech_probability(b"\x00\x00" * 512)


def test_silero_refuse_une_frequence_non_supportee() -> None:
    with pytest.raises(EngineUnavailable, match="8000 ou 16000"):
        SileroVAD(sample_rate=44100)

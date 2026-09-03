"""Fin d'énoncé : ne pas couper l'utilisateur, ne pas perdre sa première syllabe."""

from __future__ import annotations

import struct

import pytest

from lily.core.endpointer import BargeInDetector, Endpointer, EndReason
from lily.core.engines.vad.scripted import ScriptedVAD

TRAME = 512
MS_PAR_TRAME = TRAME / 16000 * 1000  # 32 ms


def trame(valeur: int = 4000) -> bytes:
    return struct.pack(f"<{TRAME}h", *([valeur, -valeur] * (TRAME // 2)))


def derouler(endpointer: Endpointer, probabilites: list[float]):
    """Pousse une suite de probabilités et rend l'énoncé produit, s'il y en a un."""
    for _ in probabilites:
        resultat = endpointer.push(trame())
        if resultat is not None:
            return resultat
    return None


def construire(probabilites: list[float], **kwargs) -> Endpointer:
    defauts = {"min_speech_ms": 100, "silence_ms": 300, "pre_roll_ms": 100,
               "min_total_speech_ms": 100, "max_wait_s": 10.0}
    defauts.update(kwargs)
    return Endpointer(ScriptedVAD(probabilites, frame_size=TRAME), **defauts)


def test_l_enonce_se_termine_sur_le_silence() -> None:
    probabilites = [0.0] * 3 + [0.9] * 10 + [0.0] * 15
    endpointer = construire(probabilites)
    enonce = derouler(endpointer, probabilites)

    assert enonce is not None and enonce
    assert enonce.reason is EndReason.SILENCE
    assert enonce.speech_ms == pytest.approx(10 * MS_PAR_TRAME, abs=1)


def test_le_pre_roll_garde_le_debut_du_mot() -> None:
    # Sans pré-roll, le temps que le VAD s'accorde sur « il parle », la
    # première syllabe est déjà passée.
    probabilites = [0.0] * 5 + [0.9] * 6 + [0.0] * 12
    endpointer = construire(probabilites, min_speech_ms=100, pre_roll_ms=200)
    enonce = derouler(endpointer, probabilites)

    assert enonce is not None
    # 6 trames de parole + 10 de silence de fin = 16 ; avec le pré-roll on en
    # garde davantage, donc l'audio dépasse la seule zone de parole.
    trames_conservees = enonce.audio.n_samples / TRAME
    assert trames_conservees > 6 + 300 / MS_PAR_TRAME


def test_une_hesitation_ne_termine_pas_la_phrase() -> None:
    # « Mets un minuteur de… [silence] …dix minutes » doit rester un énoncé.
    probabilites = [0.9] * 5 + [0.0] * 5 + [0.9] * 5 + [0.0] * 15
    endpointer = construire(probabilites, silence_ms=300)
    enonce = derouler(endpointer, probabilites)

    assert enonce is not None
    # 5 trames de silence = 160 ms < 300 ms : on n'a pas coupé au milieu.
    assert enonce.speech_ms == pytest.approx(10 * MS_PAR_TRAME, abs=1)


def test_un_enonce_interminable_finit_par_etre_coupe() -> None:
    probabilites = [0.9] * 200
    endpointer = construire(probabilites, max_utterance_s=1.0)
    enonce = derouler(endpointer, probabilites)

    assert enonce is not None
    assert enonce.reason is EndReason.TOO_LONG
    assert bool(enonce)   # tronqué, mais exploitable
    assert enonce.audio.duration_s == pytest.approx(1.0, abs=0.1)


def test_si_personne_ne_parle_on_abandonne() -> None:
    probabilites = [0.0] * 100
    endpointer = construire(probabilites, max_wait_s=0.5)
    enonce = derouler(endpointer, probabilites)

    # Attention à la nuance : l'énoncé existe (l'attente est finie) mais il
    # est faux (il n'y a rien à transcrire).
    assert enonce is not None
    assert not enonce
    assert enonce.reason is EndReason.TIMEOUT
    assert enonce.audio.n_samples == 0


def test_un_bruit_bref_n_est_pas_une_phrase() -> None:
    # Une porte qui claque : assez pour démarrer, pas assez pour transcrire.
    probabilites = [0.9] * 4 + [0.0] * 15
    endpointer = construire(probabilites, min_speech_ms=100, min_total_speech_ms=500)
    enonce = derouler(endpointer, probabilites)

    assert enonce is not None
    assert not enonce
    assert enonce.reason is EndReason.TOO_SHORT


def test_flush_termine_de_force() -> None:
    endpointer = construire([0.9] * 10)
    for _ in range(6):
        endpointer.push(trame())
    enonce = endpointer.flush()
    assert enonce.reason is EndReason.SILENCE
    assert bool(enonce)


def test_flush_sans_parole_ne_rend_rien() -> None:
    endpointer = construire([0.0] * 10)
    endpointer.push(trame())
    assert not endpointer.flush()


def test_l_endpointer_est_reutilisable() -> None:
    probabilites = [0.9] * 6 + [0.0] * 12
    endpointer = construire(probabilites * 2)
    premier = derouler(endpointer, probabilites)
    assert premier is not None and premier
    # Le VAD scripté continue sa liste : le second énoncé doit fonctionner
    # sans réinitialisation manuelle.
    endpointer.vad.reset()
    second = derouler(endpointer, probabilites)
    assert second is not None and second


# --- barge-in --------------------------------------------------------------

def test_le_barge_in_ignore_le_debut_de_la_reponse() -> None:
    # Le temps que le niveau s'établisse, le micro entend surtout le
    # haut-parleur : on ne l'écoute pas.
    detecteur = BargeInDetector(
        ScriptedVAD([1.0] * 50, frame_size=TRAME), min_speech_ms=100, guard_ms=300
    )
    # 300 ms de garde à 32 ms par trame : les neuf premières ne comptent pas,
    # même si le VAD est à 1.0 tout du long.
    assert not any(detecteur.push(trame()) for _ in range(9))


def test_le_barge_in_exige_une_parole_soutenue() -> None:
    # Un claquement isolé ne doit pas couper Lily.
    probabilites = [0.0] * 15 + [1.0, 0.0, 1.0, 0.0] * 5
    detecteur = BargeInDetector(
        ScriptedVAD(probabilites, frame_size=TRAME), min_speech_ms=200, guard_ms=100
    )
    assert not any(detecteur.push(trame()) for _ in probabilites)


def test_le_barge_in_se_declenche_sur_une_vraie_phrase() -> None:
    probabilites = [0.0] * 5 + [1.0] * 20
    detecteur = BargeInDetector(
        ScriptedVAD(probabilites, frame_size=TRAME), min_speech_ms=200, guard_ms=100
    )
    assert any(detecteur.push(trame()) for _ in probabilites)


def test_le_seuil_du_barge_in_est_plus_haut_que_celui_de_l_ecoute() -> None:
    # 0.6 suffirait à l'écoute normale, pas à interrompre : c'est ce qui
    # évite que Lily se coupe elle-même en entendant sa propre voix.
    probabilites = [0.6] * 40
    detecteur = BargeInDetector(
        ScriptedVAD(probabilites, frame_size=TRAME),
        threshold=0.85, min_speech_ms=100, guard_ms=0,
    )
    assert not any(detecteur.push(trame()) for _ in probabilites)

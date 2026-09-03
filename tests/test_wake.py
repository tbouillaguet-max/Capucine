"""Mot d'éveil : openWakeWord, repli Vosk, et la chaîne de replis du montage."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from lily.core.config import Config
from lily.core.engines.factory import build_wake
from lily.core.engines.wake.openwakeword import OpenWakeWordEngine
from lily.core.engines.wake.scripted import ScriptedWakeWord
from lily.core.engines.wake.vosk import VoskWakeWord
from lily.core.errors import EngineUnavailable

TRAME_OWW = b"\x00\x00" * 1280


# --- doublure ---------------------------------------------------------------

def test_le_moteur_scripte_se_declenche_aux_trames_prevues() -> None:
    moteur = ScriptedWakeWord([1, 3])
    resultats = [moteur.process(b"") is not None for _ in range(5)]
    assert resultats == [False, True, False, True, False]


# --- openWakeWord -----------------------------------------------------------

class _FauxModeleOww:
    scores: dict = {"lily": 0.9}
    remises_a_zero = 0
    trames: list = []

    def __init__(self, **kwargs):
        _FauxModeleOww.kwargs = kwargs

    def predict(self, x):
        _FauxModeleOww.trames.append(len(x))
        return dict(_FauxModeleOww.scores)

    def reset(self):
        _FauxModeleOww.remises_a_zero += 1


@pytest.fixture
def faux_oww(monkeypatch, tmp_path: Path):
    module = types.ModuleType("openwakeword")
    sous_module = types.ModuleType("openwakeword.model")
    sous_module.Model = _FauxModeleOww  # type: ignore[attr-defined]
    module.model = sous_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", module)
    monkeypatch.setitem(sys.modules, "openwakeword.model", sous_module)
    (tmp_path / "lily.onnx").write_bytes(b"onnx")
    _FauxModeleOww.scores = {"lily": 0.9}
    _FauxModeleOww.remises_a_zero = 0
    _FauxModeleOww.trames = []
    return tmp_path


def test_openwakeword_detecte_au_dessus_du_seuil(faux_oww) -> None:
    moteur = OpenWakeWordEngine(models_dir=faux_oww, threshold=0.5)
    assert moteur.available()
    assert moteur.frame_size == 1280

    evenement = moteur.process(TRAME_OWW)
    assert evenement is not None
    assert evenement.word == "lily"
    assert evenement.score == pytest.approx(0.9)
    # La trame arrive bien en entiers 16 bits, pas en octets.
    assert _FauxModeleOww.trames == [1280]
    # Choix assumé : onnxruntime plutôt que tflite, qui n'a pas de roue partout.
    assert _FauxModeleOww.kwargs["inference_framework"] == "onnx"


def test_openwakeword_se_tait_sous_le_seuil(faux_oww) -> None:
    _FauxModeleOww.scores = {"lily": 0.2}
    moteur = OpenWakeWordEngine(models_dir=faux_oww, threshold=0.5)
    assert moteur.process(TRAME_OWW) is None


def test_l_anti_rebond_evite_les_rafales(faux_oww) -> None:
    # Sans lui, le mot encore présent dans la fenêtre d'analyse redéclenche à
    # chaque trame suivante.
    moteur = OpenWakeWordEngine(models_dir=faux_oww, debounce_s=60.0)
    assert moteur.process(TRAME_OWW) is not None
    assert all(moteur.process(TRAME_OWW) is None for _ in range(5))
    # Et l'état interne est purgé après un déclenchement.
    assert _FauxModeleOww.remises_a_zero == 1


def test_sans_anti_rebond_chaque_trame_peut_declencher(faux_oww) -> None:
    moteur = OpenWakeWordEngine(models_dir=faux_oww, debounce_s=0.0)
    assert sum(moteur.process(TRAME_OWW) is not None for _ in range(3)) == 3


def test_le_message_renvoie_vers_le_script_d_entrainement(tmp_path: Path) -> None:
    moteur = OpenWakeWordEngine(models_dir=tmp_path)
    assert not moteur.available()
    with pytest.raises(EngineUnavailable, match="entrainer_lily"):
        moteur.process(TRAME_OWW)


# --- Vosk -------------------------------------------------------------------

class _FauxRecogniser:
    partiels: list = []

    def __init__(self, model, rate, grammar=None):
        _FauxRecogniser.grammaire = grammar
        self._index = 0

    def AcceptWaveform(self, data):  # noqa: N802 - API Vosk
        return False

    def PartialResult(self):  # noqa: N802
        texte = _FauxRecogniser.partiels[self._index] if self._index < len(_FauxRecogniser.partiels) else ""
        self._index += 1
        return json.dumps({"partial": texte})

    def Result(self):  # noqa: N802
        return json.dumps({"text": ""})


@pytest.fixture
def faux_vosk(monkeypatch, tmp_path: Path):
    module = types.ModuleType("vosk")
    module.Model = lambda chemin: ("modele", chemin)  # type: ignore[attr-defined]
    module.KaldiRecognizer = _FauxRecogniser  # type: ignore[attr-defined]
    module.SetLogLevel = lambda niveau: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vosk", module)
    (tmp_path / "modele").mkdir()
    _FauxRecogniser.partiels = []
    return tmp_path / "modele"


def test_la_grammaire_vosk_est_restreinte(faux_vosk) -> None:
    # Le décodeur ne peut littéralement rien reconnaître d'autre : c'est ce
    # qui le rend assez léger pour tourner en continu sur un Pi.
    moteur = VoskWakeWord(model_path=faux_vosk)
    assert moteur.available()
    grammaire = json.loads(moteur.grammar())
    assert "lily" in grammaire
    assert "[unk]" in grammaire


def test_vosk_detecte_le_mot_dans_un_resultat_partiel(faux_vosk) -> None:
    _FauxRecogniser.partiels = ["", "ma", "ma lily"]
    moteur = VoskWakeWord(model_path=faux_vosk)
    resultats = [moteur.process(b"\x00" * 4000) for _ in range(3)]
    assert [r is not None for r in resultats] == [False, False, True]
    assert resultats[2].word == "lily"


def test_vosk_ignore_un_mot_proche_mais_different(faux_vosk) -> None:
    _FauxRecogniser.partiels = ["la limite", "des lilas"]
    moteur = VoskWakeWord(model_path=faux_vosk)
    assert all(moteur.process(b"\x00" * 4000) is None for _ in range(2))


def test_vosk_dit_ou_trouver_son_modele(tmp_path: Path) -> None:
    moteur = VoskWakeWord(model_path=tmp_path / "absent")
    assert not moteur.available()


# --- chaîne de replis du montage -------------------------------------------

def _config(**wake) -> Config:
    return Config({"wake": {"word": "lily", **wake}})


def test_le_montage_choisit_openwakeword_quand_il_est_pret(faux_oww) -> None:
    moteur = build_wake(_config(engine="openwakeword", models_dir=str(faux_oww)))
    assert moteur is not None and moteur.name == "openwakeword"


def test_le_montage_bascule_sur_vosk_si_le_modele_n_est_pas_entraine(
    faux_vosk, tmp_path: Path
) -> None:
    # C'est l'état normal du projet tant que le modèle n'existe pas.
    moteur = build_wake(_config(
        engine="openwakeword",
        models_dir=str(tmp_path / "vide"),
        vosk_model_path=str(faux_vosk),
    ))
    assert moteur is not None and moteur.name == "vosk"


def test_sans_aucun_moteur_on_ecoute_en_permanence(tmp_path: Path) -> None:
    # Dégradé mais utilisable : mieux que refuser de démarrer.
    moteur = build_wake(_config(
        engine="openwakeword",
        models_dir=str(tmp_path / "vide"),
        vosk_model_path=str(tmp_path / "vide"),
    ))
    assert moteur is None

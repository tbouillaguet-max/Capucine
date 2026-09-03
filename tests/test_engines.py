"""Moteurs de transcription et de synthèse.

Les poids ne sont pas téléchargeables en intégration continue, donc deux
niveaux de vérification se complètent :

* des **doublures** de ``piper`` et ``faster_whisper`` injectées dans
  ``sys.modules``, qui vérifient que l'adaptateur appelle ce qu'il faut et lit
  les bons champs en retour ;
* un contrôle des **signatures réelles** des bibliothèques installées, qui
  échoue si une mise à jour change l'API sous nos pieds.
"""

from __future__ import annotations

import inspect
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from lily.core.audio import AudioBuffer
from lily.core.engines.stt.fasterwhisper import FasterWhisperSTT
from lily.core.engines.stt.scripted import ScriptedSTT
from lily.core.engines.stt.vosk import VoskSTT
from lily.core.engines.tts.piper import PiperTTS
from lily.core.engines.tts.silent import SilentTTS
from lily.core.errors import EngineUnavailable


def bruit(secondes: float = 1.0, amplitude: int = 8000, sample_rate: int = 16000) -> AudioBuffer:
    import struct

    n = int(secondes * sample_rate)
    return AudioBuffer(struct.pack(f"<{n}h", *([amplitude, -amplitude] * (n // 2))), sample_rate)


def silence(secondes: float = 1.0, sample_rate: int = 16000) -> AudioBuffer:
    return AudioBuffer(b"\x00\x00" * int(secondes * sample_rate), sample_rate)


# --- transcription factice -------------------------------------------------

def test_la_transcription_scriptee_rend_les_phrases_dans_l_ordre() -> None:
    stt = ScriptedSTT(["bonjour", "quelle heure est-il"])
    assert stt.transcribe(bruit()).text == "bonjour"
    assert stt.transcribe(bruit()).text == "quelle heure est-il"
    assert not stt.transcribe(bruit())  # à court de script : transcription vide
    assert len(stt.buffers) == 3


# --- faster-whisper --------------------------------------------------------

@dataclass
class _FauxSegment:
    text: str
    avg_logprob: float = -0.2


@dataclass
class _FauxInfo:
    language: str = "fr"


class _FauxWhisper:
    derniers_kwargs: dict = {}

    def __init__(self, taille, **kwargs):
        self.taille = taille
        self.kwargs = kwargs
        _FauxWhisper.derniers_kwargs = kwargs

    def transcribe(self, audio, **kwargs):
        _FauxWhisper.derniers_kwargs = kwargs
        texte = getattr(self, "texte", "quelle heure est-il")
        return iter([_FauxSegment(f" {texte} ")]), _FauxInfo()


@pytest.fixture
def faux_whisper(monkeypatch):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _FauxWhisper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return module


def test_whisper_transcrit_et_nettoie(faux_whisper) -> None:
    stt = FasterWhisperSTT(model="tiny")
    resultat = stt.transcribe(bruit())
    assert resultat.text == "quelle heure est-il"
    assert resultat.language == "fr"
    assert resultat.confidence == pytest.approx(0.819, abs=0.01)
    # Sur des énoncés courts et indépendants, conditionner sur le tour
    # précédent fait boucler le modèle.
    assert _FauxWhisper.derniers_kwargs["condition_on_previous_text"] is False
    assert _FauxWhisper.derniers_kwargs["language"] == "fr"


def test_whisper_n_est_pas_appele_sur_du_silence(faux_whisper) -> None:
    # Whisper invente des phrases entières sur du silence : on lui épargne
    # l'extrait, ce qui évite les commandes fantômes et gagne de la latence.
    stt = FasterWhisperSTT(model="tiny")
    assert not stt.transcribe(silence())
    assert stt._model is None  # le modèle n'a même pas été chargé


def test_whisper_ignore_les_extraits_trop_courts(faux_whisper) -> None:
    stt = FasterWhisperSTT(model="tiny", min_duration_s=0.5)
    assert not stt.transcribe(bruit(0.1))


def test_les_hallucinations_connues_sont_ecartees(faux_whisper) -> None:
    stt = FasterWhisperSTT(model="tiny")
    stt._model = _FauxWhisper("tiny")
    stt._model.texte = "Sous-titrage Société Radio-Canada"
    assert not stt.transcribe(bruit())

    stt._model.texte = "Sous-titrage Société Radio-Canada"
    permissif = FasterWhisperSTT(model="tiny", drop_hallucinations=False)
    permissif._model = stt._model
    assert permissif.transcribe(bruit()).text.startswith("Sous-titrage")


def test_auto_est_traduit_dans_le_vocabulaire_de_ctranslate2() -> None:
    assert FasterWhisperSTT(compute_type="auto").compute_type == "default"
    assert FasterWhisperSTT(compute_type="int8").compute_type == "int8"


def test_les_arguments_passes_a_whisper_existent_vraiment() -> None:
    # Garde-fou contre une dérive d'API : ce test échoue si faster-whisper
    # renomme ou retire un de ces paramètres.
    faster_whisper = pytest.importorskip("faster_whisper")
    inspect.signature(faster_whisper.WhisperModel.__init__).bind_partial(
        None, "small", device="auto", compute_type="default", cpu_threads=0, download_root=None,
    )
    inspect.signature(faster_whisper.WhisperModel.transcribe).bind_partial(
        None, object(), language="fr", beam_size=5, vad_filter=False, initial_prompt=None,
        no_speech_threshold=0.6, condition_on_previous_text=False,
    )


# --- vosk ------------------------------------------------------------------

def test_vosk_dit_ou_trouver_son_modele(tmp_path: Path) -> None:
    stt = VoskSTT(model_path=tmp_path / "absent")
    assert not stt.available()
    if "vosk" in sys.modules or _vosk_installe():
        with pytest.raises(EngineUnavailable, match="alphacephei"):
            stt.transcribe(bruit())


def _vosk_installe() -> bool:
    try:
        import vosk  # noqa: F401
    except ImportError:
        return False
    return True


# --- synthèse factice ------------------------------------------------------

def test_la_synthese_factice_rend_un_morceau_par_phrase() -> None:
    tts = SilentTTS()
    morceaux = list(tts.synthesize("Il est midi. Le soleil brille."))
    assert [m.text for m in morceaux] == ["Il est midi.", "Le soleil brille."]
    assert all(m.sample_rate == 22050 for m in morceaux)
    # La durée suit la longueur du texte : les scénarios de barge-in restent réalistes.
    assert morceaux[1].duration_s > morceaux[0].duration_s


def test_la_synthese_s_arrete_entre_deux_phrases() -> None:
    tts = SilentTTS()
    cancel = threading.Event()
    morceaux = []
    for morceau in tts.synthesize("Une. Deux. Trois.", cancel=cancel):
        morceaux.append(morceau)
        cancel.set()
    assert [m.text for m in morceaux] == ["Une."]


# --- piper -----------------------------------------------------------------

@dataclass
class _FauxMorceauPiper:
    audio_int16_bytes: bytes = b"\x01\x02"
    sample_rate: int = 22050
    sample_width: int = 2
    sample_channels: int = 1


@dataclass
class _FauxSynthesisConfig:
    speaker_id: int | None = None
    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w_scale: float | None = None
    normalize_audio: bool = True
    volume: float = 1.0


class _FauxPiperVoice:
    charge: dict = {}
    phrases: list = []

    @classmethod
    def load(cls, model_path, config_path=None, use_cuda=False, **kwargs):
        cls.charge = {"model_path": Path(model_path), "use_cuda": use_cuda}
        cls.phrases = []
        return cls()

    def synthesize(self, text, syn_config=None, include_alignments=False):
        _FauxPiperVoice.phrases.append((text, syn_config))
        return iter([_FauxMorceauPiper(), _FauxMorceauPiper(b"\x03\x04")])


@pytest.fixture
def faux_piper(monkeypatch, tmp_path: Path):
    module = types.ModuleType("piper")
    module.PiperVoice = _FauxPiperVoice  # type: ignore[attr-defined]
    module.SynthesisConfig = _FauxSynthesisConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", module)
    (tmp_path / "fr_FR-siwis-medium.onnx").write_bytes(b"onnx")
    return tmp_path


def test_piper_rend_un_morceau_par_phrase(faux_piper) -> None:
    tts = PiperTTS(models_dir=faux_piper)
    assert tts.available()
    morceaux = list(tts.synthesize("Il est midi. Le soleil brille."))

    assert [m.text for m in morceaux] == ["Il est midi.", "Le soleil brille."]
    # Les morceaux d'une même phrase sont recollés, la fréquence est reprise.
    assert morceaux[0].pcm == b"\x01\x02\x03\x04"
    assert morceaux[0].sample_rate == 22050
    assert [texte for texte, _ in _FauxPiperVoice.phrases] == [
        "Il est midi.", "Le soleil brille.",
    ]


def test_la_vitesse_est_l_inverse_de_l_echelle_de_duree(faux_piper) -> None:
    # Piper raisonne en durée ; nous exposons une vitesse.
    list(PiperTTS(models_dir=faux_piper, speed=2.0).synthesize("Bonjour."))
    _, config = _FauxPiperVoice.phrases[-1]
    assert config.length_scale == pytest.approx(0.5)

    list(PiperTTS(models_dir=faux_piper, speed=1.0).synthesize("Bonjour."))
    _, config = _FauxPiperVoice.phrases[-1]
    assert config.length_scale is None


def test_piper_s_arrete_entre_deux_phrases(faux_piper) -> None:
    tts = PiperTTS(models_dir=faux_piper)
    cancel = threading.Event()
    cancel.set()
    assert list(tts.synthesize("Une. Deux.", cancel=cancel)) == []


def test_piper_dit_comment_telecharger_la_voix_manquante(tmp_path: Path) -> None:
    tts = PiperTTS(models_dir=tmp_path, voice="fr_FR-siwis-medium")
    assert not tts.available()
    with pytest.raises(EngineUnavailable, match="download_voices"):
        list(tts.synthesize("Bonjour."))


def test_les_arguments_passes_a_piper_existent_vraiment() -> None:
    piper = pytest.importorskip("piper")
    inspect.signature(piper.PiperVoice.load).bind_partial(
        Path("voix.onnx"), use_cuda=False,
    )
    inspect.signature(piper.PiperVoice.synthesize).bind_partial(
        None, "Bonjour.", syn_config=None,
    )
    import dataclasses

    champs = {champ.name for champ in dataclasses.fields(piper.SynthesisConfig)}
    assert {"speaker_id", "length_scale", "noise_scale", "noise_w_scale", "volume"} <= champs

    # `sample_rate` est un champ de dataclasse, `audio_int16_bytes` une propriété.
    morceau = {champ.name for champ in dataclasses.fields(piper.AudioChunk)} | set(dir(piper.AudioChunk))
    assert {"audio_int16_bytes", "sample_rate"} <= morceau


# --- « injoignable » recouvre deux pannes très différentes -------------------

class _SansPaquet:
    """Fait disparaître un module, comme sur une machine où il n'est pas installé."""

    def __init__(self, monkeypatch, nom: str) -> None:
        import builtins
        import sys

        vrai = builtins.__import__

        def faux(module, *args, **kwargs):
            if module == nom or module.startswith(f"{nom}."):
                raise ImportError(f"No module named '{nom}'")
            return vrai(module, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", faux)
        monkeypatch.setitem(sys.modules, nom, None)
        monkeypatch.delitem(sys.modules, nom, raising=False)


def test_un_paquet_absent_ne_se_confond_pas_avec_un_service_muet(monkeypatch) -> None:
    """La panne que l'utilisateur a réellement rencontrée.

    « ollama pull » marche par le binaire natif ; le paquet Python est une
    dépendance séparée. Annoncer « injoignable » quand c'est lui qui manque
    envoie réinstaller un service qui tourne très bien.
    """
    from lily.core.engines.llm.ollama import OllamaLLM

    _SansPaquet(monkeypatch, "ollama")
    moteur = OllamaLLM()
    assert not moteur.available()
    assert "paquet" in moteur.unavailable_reason()
    assert "pip install ollama" in moteur.unavailable_reason()


def test_un_service_muet_dit_ou_le_chercher() -> None:
    from lily.core.engines.llm.ollama import OllamaLLM

    # Port absurde : rien n'écoute, mais le paquet est là.
    moteur = OllamaLLM(host="http://127.0.0.1:1")
    assert not moteur.available()
    raison = moteur.unavailable_reason()
    assert "ne répond pas" in raison and "127.0.0.1:1" in raison


def test_le_vectoriseur_distingue_aussi_les_deux(monkeypatch) -> None:
    from lily.core.engines.embeddings.ollama import OllamaEmbeddings

    muet = OllamaEmbeddings(host="http://127.0.0.1:1")
    assert not muet.available()
    assert "ne répond pas" in muet.unavailable_reason()

    _SansPaquet(monkeypatch, "ollama")
    sans_paquet = OllamaEmbeddings()
    assert not sans_paquet.available()
    assert "pip install ollama" in sans_paquet.unavailable_reason()


def test_le_modele_non_tire_est_une_troisieme_panne(monkeypatch) -> None:
    from lily.core.engines.embeddings.ollama import OllamaEmbeddings

    class Entree:
        model = "qwen2.5:7b-instruct-q4_K_M"

    class Listing:
        models = [Entree()]

    moteur = OllamaEmbeddings()
    monkeypatch.setattr(
        moteur, "_get_client",
        lambda: type("C", (), {"list": staticmethod(lambda: Listing())})(),
    )
    assert not moteur.available()
    raison = moteur.unavailable_reason()
    # Le service répond : ce n'est ni le paquet, ni le service.
    assert "ollama pull nomic-embed-text" in raison
    assert "répond" in raison


def test_un_moteur_disponible_n_a_pas_de_raison(monkeypatch) -> None:
    from lily.core.engines.embeddings.ollama import OllamaEmbeddings

    class Entree:
        model = "nomic-embed-text:latest"

    class Listing:
        models = [Entree()]

    moteur = OllamaEmbeddings()
    monkeypatch.setattr(
        moteur, "_get_client",
        lambda: type("C", (), {"list": staticmethod(lambda: Listing())})(),
    )
    assert moteur.available()
    assert moteur.unavailable_reason() == ""


def test_llamacpp_nomme_ce_qui_manque(tmp_path) -> None:
    from lily.core.engines.llm.llamacpp import LlamaCppLLM

    moteur = LlamaCppLLM()
    assert not moteur.available()
    assert "llm.model_path" in moteur.unavailable_reason()

"""Le corpus d'éveil : garder la bonne fenêtre, et rien d'autre."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from capucine.core.config import Config
from capucine.core.corpus import CorpusEveil
from capucine.core.corpus import depuis_config as corpus_depuis_config
from capucine.core.endpointer import Endpointer
from capucine.core.interfaces.wake import WakeEvent, WakeWordEngine
from capucine.core.listener import ListenerEvent, ListenMode, VoiceListener

TRAME = b"\x11\x22" * 480          # 30 ms à 16 kHz


@pytest.fixture
def corpus(tmp_path: Path) -> CorpusEveil:
    return CorpusEveil(
        tmp_path / "corpus", actif=True,
        secondes_avant=0.3, secondes_apres=0.06, sample_rate=16000,
    )


def _duree(chemin: Path) -> float:
    with wave.open(str(chemin), "rb") as fichier:
        return fichier.getnframes() / fichier.getframerate()


# --- l'anneau ----------------------------------------------------------------

def test_eteint_il_n_ecrit_rien(tmp_path: Path) -> None:
    muet = CorpusEveil(tmp_path / "c", actif=False)
    for _ in range(50):
        muet.alimenter(TRAME)
    muet.declencher(0.9)
    muet.cloturer()
    assert muet.etat().total == 0
    assert not (tmp_path / "c").exists()      # pas même un dossier créé


def test_l_anneau_ne_garde_que_la_fenetre_demandee(corpus: CorpusEveil) -> None:
    # Dix secondes de son pour une fenêtre de 0,3 : le reste doit disparaître.
    for _ in range(333):
        corpus.alimenter(TRAME)
    corpus.declencher(0.9)
    corpus.cloturer()
    (extrait,) = list((corpus.dossier / "en_attente").glob("*.wav"))
    assert _duree(extrait) == pytest.approx(0.3, abs=0.05)


def test_la_queue_prolonge_puis_ecrit_toute_seule(corpus: CorpusEveil) -> None:
    for _ in range(20):
        corpus.alimenter(TRAME)
    corpus.declencher(0.9)
    assert not list((corpus.dossier / "en_attente").glob("*.wav"))
    # 60 ms de queue : deux trames de 30 ms suffisent.
    corpus.completer(TRAME)
    corpus.completer(TRAME)
    (extrait,) = list((corpus.dossier / "en_attente").glob("*.wav"))
    assert _duree(extrait) == pytest.approx(0.36, abs=0.05)


def test_completer_n_enregistre_jamais_hors_declenchement(corpus: CorpusEveil) -> None:
    # C'est la garantie qui compte : ce que vous dites APRÈS le mot d'éveil ne
    # doit jamais entrer dans le corpus.
    for _ in range(50):
        corpus.completer(TRAME)
    corpus.declencher(0.9)
    corpus.cloturer()
    assert not list((corpus.dossier / "en_attente").glob("*.wav"))


def test_un_second_declenchement_n_ecrase_pas_le_premier(corpus: CorpusEveil) -> None:
    for _ in range(20):
        corpus.alimenter(TRAME)
    corpus.declencher(0.9)
    corpus.declencher(0.4)      # ignoré : une capture est déjà en cours
    corpus.cloturer()
    assert len(list((corpus.dossier / "en_attente").glob("*.wav"))) == 1


# --- l'étiquetage ------------------------------------------------------------

def _capturer(corpus: CorpusEveil) -> None:
    for _ in range(20):
        corpus.alimenter(TRAME)
    corpus.declencher(0.87)
    corpus.cloturer()


def test_un_eveil_confirme_devient_un_positif(corpus: CorpusEveil) -> None:
    _capturer(corpus)
    chemin = corpus.confirmer()
    assert chemin is not None and chemin.parent.name == "eveils"
    etat = corpus.etat()
    assert (etat.eveils, etat.faux_positifs, etat.en_attente) == (1, 0, 0)


def test_un_eveil_dementi_devient_un_negatif(corpus: CorpusEveil) -> None:
    _capturer(corpus)
    chemin = corpus.dementir()
    assert chemin is not None and chemin.parent.name == "faux_positifs"
    assert corpus.etat().taux_de_faux_positifs == 1.0


def test_le_score_est_dans_le_nom_du_fichier(corpus: CorpusEveil) -> None:
    _capturer(corpus)
    chemin = corpus.confirmer()
    assert chemin.stem.endswith("_s087")


def test_etiqueter_sans_capture_ne_casse_rien(corpus: CorpusEveil) -> None:
    assert corpus.confirmer() is None
    assert corpus.dementir() is None


def test_on_n_etiquette_pas_deux_fois_le_meme_extrait(corpus: CorpusEveil) -> None:
    _capturer(corpus)
    assert corpus.confirmer() is not None
    assert corpus.dementir() is None
    etat = corpus.etat()
    assert (etat.eveils, etat.faux_positifs) == (1, 0)


def test_le_corpus_est_borne(tmp_path: Path) -> None:
    petit = CorpusEveil(
        tmp_path / "c", actif=True, secondes_avant=0.3, secondes_apres=0.03,
        maximum_par_classe=3,
    )
    for _ in range(6):
        _capturer(petit)
        petit.confirmer()
    # Un corpus d'éveil ne doit pas devenir une archive du salon.
    assert petit.etat().eveils == 3


def test_oublier_les_en_attente(corpus: CorpusEveil) -> None:
    _capturer(corpus)
    _capturer(corpus)          # le premier reste orphelin en attente
    assert corpus.oublier_les_en_attente() >= 1
    assert corpus.etat().en_attente == 0


def test_tout_oublier(corpus: CorpusEveil) -> None:
    _capturer(corpus)
    corpus.confirmer()
    _capturer(corpus)
    corpus.dementir()
    assert corpus.tout_oublier() == 2
    assert corpus.etat().total == 0


# --- le fil d'écoute l'alimente ---------------------------------------------

class EveilScripte(WakeWordEngine):
    """Déclenche à la n-ième trame, sans modèle."""

    name = "scripte"

    def __init__(self, a_la_trame: int = 5) -> None:
        self.a_la_trame = a_la_trame
        self.vues = 0

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_size(self) -> int:
        return 480

    def available(self) -> bool:
        return True

    def process(self, frame: bytes) -> WakeEvent | None:
        self.vues += 1
        if self.vues == self.a_la_trame:
            return WakeEvent(word="capucine", score=0.91, timestamp=0.0)
        return None


def test_le_fil_d_ecoute_nourrit_le_corpus(corpus: CorpusEveil, tmp_path: Path) -> None:
    from capucine.core.audio import MemoryAudioInput
    from capucine.core.engines.vad.scripted import ScriptedVAD

    evenements: list[ListenerEvent] = []
    # Douze trames de 30 ms avant le déclenchement : de quoi remplir la
    # fenêtre au-delà du minimum en deçà duquel un extrait est jeté.
    mic = MemoryAudioInput([TRAME] * 30, sample_rate=16000, frame_size=480)
    listener = VoiceListener(
        mic,
        endpointer=Endpointer(ScriptedVAD([0.0] * 40)),
        on_event=evenements.append,
        wake=EveilScripte(a_la_trame=13),
        corpus=corpus,
        start_mode=ListenMode.WAKE,
    )
    listener.start()
    listener.stop(timeout=3.0)

    assert any(evenement.kind == "wake" for evenement in evenements)
    corpus.cloturer()
    (extrait,) = list((corpus.dossier / "en_attente").glob("*.wav"))
    # Les trames qui précédaient le déclenchement ont bien été gardées.
    assert _duree(extrait) >= 0.2


# --- configuration -----------------------------------------------------------

def test_le_corpus_est_eteint_par_defaut(tmp_path: Path) -> None:
    config = Config({"corpus": {"dossier": str(tmp_path / "c")}})
    magasin = corpus_depuis_config(config)
    # Écrire du son sur un disque est une décision de l'utilisateur, pas un
    # défaut qu'il découvrirait après coup.
    assert magasin is not None and not magasin.actif


def test_le_corpus_s_allume_par_la_configuration(tmp_path: Path) -> None:
    config = Config({"corpus": {"actif": True, "dossier": str(tmp_path / "c")}})
    magasin = corpus_depuis_config(config)
    assert magasin.actif
    assert magasin.dossier == tmp_path / "c"

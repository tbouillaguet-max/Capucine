"""Reconnaître VOTRE voix : signer, enrôler, refuser, suivre.

Les voix de ce fichier sont synthétiques et construites pour être
discriminables : une fondamentale, ses harmoniques, et trois formants qui
colorent le spectre. Ce n'est pas une mesure de la qualité du moteur sur des
voix réelles — cela demanderait un corpus enregistré — mais c'est ce qu'il
faut pour éprouver la MÉCANIQUE : ce qui est signé, ce qui est retenu, ce qui
est refusé, et ce qui ne doit jamais bloquer.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from lily.core.audio import AudioBuffer
from lily.core.config import Config
from lily.core.engines.factory import build_speaker
from lily.core.engines.locuteur.spectral import SpectralSpeaker
from lily.core.interfaces.locuteur import SpeakerEngine
from lily.core.locuteur import Locuteur
from lily.core.locuteur import depuis_config as locuteur_depuis_config

numpy = pytest.importorskip("numpy")

# Deux timbres nettement distincts : grave et formants bas contre aigu et
# formants hauts. C'est le cas que le moteur spectral doit savoir trancher.
TOM = (112.0, [(520, 90), (1100, 130), (2500, 200)])
INVITE = (205.0, [(800, 110), (1700, 160), (3100, 220)])


def voix(profil, secondes: float = 2.0, graine: int = 0, sr: int = 16000) -> AudioBuffer:
    """Une voyelle tenue, avec vibrato : deux extraits ne sont jamais identiques."""
    f0, formants = profil
    rng = numpy.random.default_rng(graine)
    t = numpy.arange(int(sr * secondes)) / sr
    derive = rng.standard_normal(t.size).cumsum() / max(1.0, t.size**0.5)
    f = f0 * (1 + 0.02 * numpy.sin(2 * numpy.pi * 4 * t) + 0.01 * derive)
    phase = 2 * numpy.pi * numpy.cumsum(f) / sr
    signal = sum(numpy.sin(k * phase) / k for k in range(1, 25))

    spectre = numpy.fft.rfft(signal)
    hz = numpy.fft.rfftfreq(signal.size, 1 / sr)
    enveloppe = numpy.full_like(hz, 0.05)
    for centre, largeur in formants:
        enveloppe += numpy.exp(-(((hz - centre) / largeur) ** 2))
    signal = numpy.fft.irfft(spectre * enveloppe, n=signal.size)
    signal += 0.01 * rng.standard_normal(signal.size)
    signal = signal / numpy.abs(signal).max() * 0.6
    pcm = (signal * 32767).astype(numpy.int16)
    return AudioBuffer(struct.pack(f"<{pcm.size}h", *pcm), sr)


def silence(secondes: float = 2.0, sr: int = 16000) -> AudioBuffer:
    return AudioBuffer(b"\x00\x00" * int(sr * secondes), sr)


def magasin(tmp_path: Path, **kwargs) -> Locuteur:
    options = {"actif": True, "echantillons_min": 2, "seuil": 0.62}
    options.update(kwargs)
    return Locuteur(tmp_path / "voix.sqlite", SpectralSpeaker(), **options)


# --- la signature -----------------------------------------------------------

def test_la_signature_est_stable_pour_une_meme_voix() -> None:
    moteur = SpectralSpeaker()
    a = numpy.array(moteur.encode(voix(TOM, graine=1)))
    b = numpy.array(moteur.encode(voix(TOM, graine=2)))
    assert a.size == moteur.dimension
    assert float(a @ b) > 0.9


def test_la_signature_distingue_deux_voix() -> None:
    moteur = SpectralSpeaker()
    tom = numpy.array(moteur.encode(voix(TOM, graine=1)))
    invite = numpy.array(moteur.encode(voix(INVITE, graine=1)))
    assert float(tom @ invite) < 0.8


def test_on_ne_signe_pas_du_silence() -> None:
    """Une signature tirée du silence serait celle du bruit de fond, et elle
    empoisonnerait la référence en s'y ajoutant."""
    assert SpectralSpeaker().encode(silence()) == []
    assert SpectralSpeaker().encode(AudioBuffer(b"", 16000)) == []
    assert SpectralSpeaker().encode(voix(TOM, secondes=0.1)) == []


# --- enrôlement -------------------------------------------------------------

def test_l_enrolement_demande_plusieurs_echantillons(tmp_path: Path) -> None:
    voix_de_tom = magasin(tmp_path, echantillons_min=3)
    assert not voix_de_tom.enrole

    premier = voix_de_tom.enroler(voix(TOM, graine=1))
    assert premier and "encore 2" in premier.raison
    voix_de_tom.enroler(voix(TOM, graine=2))
    assert not voix_de_tom.enrole

    dernier = voix_de_tom.enroler(voix(TOM, graine=3))
    assert dernier and voix_de_tom.enrole
    assert voix_de_tom.etat().echantillons == 3
    voix_de_tom.fermer()


def test_l_enrolement_reprend_la_derniere_capture(tmp_path: Path) -> None:
    """« apprends ma voix » ne fait pas répéter : c'est ce que vous venez de
    dire qui sert d'échantillon."""
    voix_de_tom = magasin(tmp_path)
    voix_de_tom.derniere_capture = voix(TOM, graine=1)
    assert voix_de_tom.enroler()
    assert voix_de_tom.etat().echantillons == 1
    voix_de_tom.fermer()


def test_un_extrait_sans_parole_est_refuse_avec_une_raison(tmp_path: Path) -> None:
    voix_de_tom = magasin(tmp_path)
    verdict = voix_de_tom.enroler(silence())
    assert not verdict
    assert "pas assez de parole" in verdict.raison
    assert voix_de_tom.etat().echantillons == 0
    voix_de_tom.fermer()


# --- reconnaissance ---------------------------------------------------------

def test_elle_reconnait_la_voix_apprise(tmp_path: Path) -> None:
    voix_de_tom = magasin(tmp_path)
    for graine in (1, 2, 3):
        voix_de_tom.enroler(voix(TOM, graine=graine))

    verdict = voix_de_tom.reconnait(voix(TOM, graine=9))
    assert verdict and verdict.score > voix_de_tom.seuil
    voix_de_tom.fermer()


def test_elle_refuse_une_autre_voix(tmp_path: Path) -> None:
    """Le cœur de la demande : un « Lily » qui ne vient pas de vous."""
    voix_de_tom = magasin(tmp_path)
    for graine in (1, 2, 3):
        voix_de_tom.enroler(voix(TOM, graine=graine))

    verdict = voix_de_tom.reconnait(voix(INVITE, graine=9))
    assert not verdict
    assert verdict.score < voix_de_tom.seuil
    assert "n'est pas celle" in verdict.raison
    voix_de_tom.fermer()


# --- les garde-fous ---------------------------------------------------------

def test_sans_enrolement_le_filtre_ne_bloque_personne(tmp_path: Path) -> None:
    """Refuser sans savoir reconnaître serait la pire des façons d'échouer."""
    voix_de_tom = magasin(tmp_path)
    assert not voix_de_tom.opere
    assert voix_de_tom.reconnait(voix(INVITE, graine=1))
    voix_de_tom.fermer()


def test_eteint_le_filtre_ne_bloque_personne(tmp_path: Path) -> None:
    voix_de_tom = magasin(tmp_path, actif=False)
    voix_de_tom.enroler(voix(TOM, graine=1))
    voix_de_tom.enroler(voix(TOM, graine=2))
    assert not voix_de_tom.opere
    assert voix_de_tom.reconnait(voix(INVITE, graine=1))
    voix_de_tom.fermer()


def test_sans_moteur_le_filtre_ne_bloque_personne(tmp_path: Path) -> None:
    voix_de_tom = Locuteur(tmp_path / "v.sqlite", None, actif=True, echantillons_min=1)
    assert not voix_de_tom.opere
    assert voix_de_tom.reconnait(voix(TOM, graine=1))
    verdict = voix_de_tom.enroler(voix(TOM, graine=1))
    assert not verdict and "aucun moteur" in verdict.raison
    voix_de_tom.fermer()


def test_un_moteur_qui_leve_laisse_passer(tmp_path: Path) -> None:
    """Une comparaison impossible ne doit pas rendre Lily sourde à son
    propriétaire : dans le doute, on ouvre."""

    class MoteurCasse(SpeakerEngine):
        name = "casse"
        appels = 0

        def available(self) -> bool:
            return True

        def encode(self, audio: AudioBuffer) -> list[float]:
            MoteurCasse.appels += 1
            if MoteurCasse.appels > 2:      # les deux premiers servent à enrôler
                raise RuntimeError("le moteur a lâché")
            return [1.0, 0.0, 0.0]

    voix_de_tom = Locuteur(
        tmp_path / "v.sqlite", MoteurCasse(), actif=True, echantillons_min=2
    )
    voix_de_tom.enroler(voix(TOM, graine=1))
    voix_de_tom.enroler(voix(TOM, graine=2))
    assert voix_de_tom.opere

    verdict = voix_de_tom.reconnait(voix(TOM, graine=3))
    assert verdict and "impossible" in verdict.raison
    voix_de_tom.fermer()


def test_oublier_rend_le_filtre_inerte(tmp_path: Path) -> None:
    voix_de_tom = magasin(tmp_path)
    voix_de_tom.enroler(voix(TOM, graine=1))
    voix_de_tom.enroler(voix(TOM, graine=2))
    assert voix_de_tom.opere

    assert voix_de_tom.oublier() == 2
    assert not voix_de_tom.opere
    assert voix_de_tom.reconnait(voix(INVITE, graine=1))
    voix_de_tom.fermer()


# --- apprentissage continu --------------------------------------------------

def test_une_voix_franchement_reconnue_enrichit_la_reference(tmp_path: Path) -> None:
    voix_de_tom = magasin(tmp_path, marge_d_apprentissage=0.0)
    for graine in (1, 2):
        voix_de_tom.enroler(voix(TOM, graine=graine))
    avant = voix_de_tom.etat().echantillons

    assert voix_de_tom.reconnait(voix(TOM, graine=7))
    assert voix_de_tom.etat().echantillons == avant + 1
    voix_de_tom.fermer()


def test_une_voix_reconnue_de_justesse_n_enrichit_rien(tmp_path: Path) -> None:
    """Le garde-fou contre la dérive.

    Sans marge, chaque voix qui passe de justesse tirerait la référence vers
    elle — et de proche en proche le filtre finirait par accepter tout le
    monde. C'est le seul mode d'échec vraiment dangereux ici, parce qu'il est
    silencieux : la reconnaissance a l'air de marcher jusqu'au jour où elle
    n'écarte plus rien.
    """
    voix_de_tom = magasin(tmp_path, seuil=0.0, marge_d_apprentissage=1.5)
    for graine in (1, 2):
        voix_de_tom.enroler(voix(TOM, graine=graine))
    avant = voix_de_tom.etat().echantillons

    assert voix_de_tom.reconnait(voix(INVITE, graine=7))   # passe : seuil à zéro
    assert voix_de_tom.etat().echantillons == avant, "la référence a dérivé"
    voix_de_tom.fermer()


def test_l_apprentissage_continu_se_coupe(tmp_path: Path) -> None:
    voix_de_tom = magasin(tmp_path, apprentissage_continu=False, marge_d_apprentissage=0.0)
    for graine in (1, 2):
        voix_de_tom.enroler(voix(TOM, graine=graine))
    avant = voix_de_tom.etat().echantillons

    assert voix_de_tom.reconnait(voix(TOM, graine=7))
    assert voix_de_tom.etat().echantillons == avant
    voix_de_tom.fermer()


def test_l_elagage_epargne_les_echantillons_donnes(tmp_path: Path) -> None:
    """Ce que vous avez donné volontairement reste ; ce qui a été ramassé à
    l'usage part en premier."""
    voix_de_tom = magasin(
        tmp_path, echantillons_max=4, marge_d_apprentissage=0.0, echantillons_min=2
    )
    for graine in (1, 2, 3):
        voix_de_tom.enroler(voix(TOM, graine=graine))
    for graine in range(10, 16):
        voix_de_tom.reconnait(voix(TOM, graine=graine))

    lignes = voix_de_tom._db.execute(
        "SELECT origine, COUNT(*) FROM empreintes_vocales GROUP BY origine"
    ).fetchall()
    compte = {ligne[0]: ligne[1] for ligne in lignes}
    assert compte["enrolement"] == 3, "un échantillon donné a été effacé"
    assert sum(compte.values()) <= 4
    voix_de_tom.fermer()


# --- montage ----------------------------------------------------------------

def test_la_fabrique_ne_monte_rien_quand_c_est_eteint() -> None:
    assert build_speaker(Config({"voix": {"actif": False}})) is None
    assert build_speaker(Config({})) is None


def test_la_fabrique_monte_le_moteur_spectral() -> None:
    moteur = build_speaker(Config({"voix": {"actif": True}}))
    assert moteur is not None and moteur.name == "spectral"


def test_un_modele_onnx_absent_ne_fait_pas_echouer_le_montage(tmp_path: Path) -> None:
    """Comme partout : un moteur absent est un état, pas une erreur."""
    config = Config({"voix": {"actif": True, "engine": "onnx",
                              "model_path": str(tmp_path / "absent.onnx")}})
    assert build_speaker(config) is None


def test_la_section_absente_donne_un_magasin_inerte(tmp_path: Path) -> None:
    """Rendu même éteint, pour qu'une compétence puisse expliquer comment
    l'allumer — même choix que pour le corpus d'éveil."""
    magasin_inerte = locuteur_depuis_config(
        Config({"memoire": {"fichier": str(tmp_path / "m.sqlite")}})
    )
    assert magasin_inerte is not None
    assert not magasin_inerte.actif
    assert not magasin_inerte.opere
    assert "éteinte" in magasin_inerte.etat().decrire()
    magasin_inerte.fermer()

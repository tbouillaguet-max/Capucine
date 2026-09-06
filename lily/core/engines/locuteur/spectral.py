"""Signature vocale par le spectre, sans autre dépendance que numpy.

Ce qui distingue deux voix tient pour l'essentiel à deux choses : la **forme
du conduit vocal**, qui colore le spectre de façon stable d'une phrase à
l'autre, et la **hauteur**, qui varie d'une personne à l'autre bien plus
qu'elle ne varie chez la même. On mesure les deux.

La recette, classique et sans surprise :

1. découper en fenêtres de 25 ms, toutes les 10 ms, fenêtrées en Hann ;
2. ne garder que les fenêtres **voisées** — le silence et le souffle ne disent
   rien de qui parle, et les inclure dilue la signature ;
3. passer au spectre de puissance, puis dans un banc de filtres mel ;
4. en tirer des MFCC par transformée en cosinus ;
5. rendre la **moyenne et l'écart-type** de chaque coefficient, plus la
   hauteur médiane et sa dispersion.

Une décision mérite d'être expliquée, parce qu'elle est contre-intuitive pour
qui vient de la reconnaissance de parole : on **ne** normalise **pas** la
moyenne des MFCC. En transcription, retrancher cette moyenne est un réflexe —
elle porte le canal, le micro, la pièce, tout ce dont on veut se débarrasser.
Ici c'est exactement l'inverse : cette moyenne porte aussi le timbre, et le
timbre est le signal. On la garde, en assumant la contrepartie — la signature
est liée à son micro et à sa pièce, et changer de matériel demande de
réenrôler.

Ce que ce moteur sait faire, et ce qu'il ne sait pas
---------------------------------------------------
Il sépare des voix de timbres et de hauteurs différents, dans un décor
acoustique stable. C'est ce qu'on lui demande : distinguer VOUS d'une
télévision, d'un invité, d'une conversation à côté.

Il ne sépare pas deux voix proches — deux frères, deux collègues de même
registre. Pour cela il faut un vrai modèle de locuteur, et c'est ce que
propose le moteur ``onnx``. Le dire ici évite de découvrir la limite le jour
où elle compte.
"""

from __future__ import annotations

import math
from typing import Any

from ...audio import AudioBuffer
from ...interfaces.locuteur import SpeakerEngine
from ...logging import get_logger

logger = get_logger("locuteur.spectral")

FENETRE_MS = 25.0
PAS_MS = 10.0
FILTRES_MEL = 26
COEFFICIENTS = 13
# En deçà, une signature ne veut rien dire : trop peu de fenêtres voisées pour
# que la moyenne soit autre chose que du bruit.
FENETRES_MINIMALES = 25
# Bornes de recherche de la hauteur, en hertz. Elles couvrent une voix grave
# d'homme comme une voix aiguë d'enfant, sans aller chercher des harmoniques.
F0_MIN = 70.0
F0_MAX = 350.0


class SpectralSpeaker(SpeakerEngine):
    """Signature spectrale. Toujours disponible dès que numpy l'est."""

    name = "spectral"

    def __init__(
        self,
        sample_rate: int = 16000,
        coefficients: int = COEFFICIENTS,
        filtres: int = FILTRES_MEL,
        **_ignored: Any,
    ) -> None:
        self.sample_rate = sample_rate
        self.coefficients = coefficients
        self.filtres = filtres
        self.model = f"{coefficients}c"
        self._banc: Any = None

    @property
    def dimension(self) -> int:
        # moyenne + écart-type par coefficient, plus deux mesures de hauteur.
        return 2 * self.coefficients + 2

    def available(self) -> bool:
        try:
            import numpy  # noqa: F401
        except ImportError:
            return False
        return True

    def unavailable_reason(self) -> str:
        if not self.available():
            return (
                "Le paquet « numpy » est absent. Il vient avec la chaîne audio : "
                'pip install -e ".[audio]"'
            )
        return ""

    def encode(self, audio: AudioBuffer) -> list[float]:
        try:
            import numpy as np
        except ImportError:
            logger.debug("numpy absent : aucune signature vocale possible.")
            return []
        if not audio or audio.duration_s < 0.3:
            return []

        echantillons = audio.to_float32()
        taille = int(self.sample_rate * FENETRE_MS / 1000)
        pas = int(self.sample_rate * PAS_MS / 1000)
        if echantillons.size < taille:
            return []

        # Les fenêtres, en une seule vue : `as_strided` évite de recopier le
        # signal autant de fois qu'il y a de fenêtres.
        nombre = 1 + (echantillons.size - taille) // pas
        fenetres = np.lib.stride_tricks.as_strided(
            echantillons,
            shape=(nombre, taille),
            strides=(echantillons.strides[0] * pas, echantillons.strides[0]),
        ) * np.hanning(taille)

        energies = np.sqrt((fenetres**2).mean(axis=1) + 1e-12)
        # Le seuil est relatif au maximum de l'extrait : un enregistrement
        # faible ne doit pas être rejeté en bloc, seulement ses creux.
        voisees = energies > max(energies.max() * 0.15, 1e-4)
        if int(voisees.sum()) < FENETRES_MINIMALES:
            logger.debug("Trop peu de parole pour signer (%d fenêtres).", int(voisees.sum()))
            return []
        retenues = fenetres[voisees]

        spectre = np.abs(np.fft.rfft(retenues, axis=1)) ** 2
        mel = np.log(spectre @ self._banc_de_filtres(np, taille).T + 1e-10)
        cepstre = _dct(np, mel)[:, : self.coefficients]

        signature = np.concatenate([
            cepstre.mean(axis=0),
            cepstre.std(axis=0),
            self._hauteur(np, retenues),
        ])
        norme = float(np.linalg.norm(signature))
        if norme == 0.0 or not np.isfinite(norme):
            return []
        return (signature / norme).astype(float).tolist()

    # -- interne ------------------------------------------------------------
    def _banc_de_filtres(self, np: Any, taille: int) -> Any:
        """Le banc mel, construit une fois pour toutes."""
        if self._banc is not None and self._banc.shape[1] == taille // 2 + 1:
            return self._banc
        haut = self.sample_rate / 2
        bornes = np.linspace(_en_mel(0.0), _en_mel(haut), self.filtres + 2)
        hertz = np.array([_en_hertz(m) for m in bornes])
        indices = np.floor((taille + 1) * hertz / self.sample_rate).astype(int)
        banc = np.zeros((self.filtres, taille // 2 + 1))
        for filtre in range(self.filtres):
            gauche, sommet, droite = indices[filtre : filtre + 3]
            for k in range(gauche, min(sommet, banc.shape[1])):
                if sommet > gauche:
                    banc[filtre, k] = (k - gauche) / (sommet - gauche)
            for k in range(sommet, min(droite, banc.shape[1])):
                if droite > sommet:
                    banc[filtre, k] = (droite - k) / (droite - sommet)
        self._banc = banc
        return banc

    def _hauteur(self, np: Any, fenetres: Any) -> Any:
        """Médiane et dispersion de la hauteur, par autocorrélation.

        La hauteur est le second grand discriminant entre deux voix, et il est
        presque orthogonal au timbre : deux personnes peuvent avoir un spectre
        proche et des hauteurs très différentes. Deux nombres suffisent —
        décrire tout le contour serait décrire la phrase, pas la personne.
        """
        decalage_min = int(self.sample_rate / F0_MAX)
        decalage_max = int(self.sample_rate / F0_MIN)
        if fenetres.shape[1] <= decalage_max:
            return np.zeros(2)

        # Autocorrélation par la FFT : bien plus rapide qu'un produit décalé.
        taille = 1 << int(fenetres.shape[1] * 2 - 1).bit_length()
        spectre = np.fft.rfft(fenetres, n=taille, axis=1)
        correlation = np.fft.irfft(spectre * np.conjugate(spectre), n=taille, axis=1)
        tranche = correlation[:, decalage_min:decalage_max]
        if tranche.size == 0:
            return np.zeros(2)

        decalages = tranche.argmax(axis=1) + decalage_min
        # Une fenêtre dont le pic d'autocorrélation est faible n'est pas
        # voisée : sa « hauteur » serait celle du bruit.
        force = tranche.max(axis=1) / (correlation[:, 0] + 1e-10)
        retenus = decalages[force > 0.3]
        if retenus.size < 3:
            return np.zeros(2)
        f0 = self.sample_rate / retenus
        # Ramenées à l'échelle des MFCC, sinon deux cents hertz écraseraient
        # tout le reste de la signature.
        return np.array([np.median(f0) / 100.0, np.std(f0) / 100.0])


def _dct(np: Any, matrice: Any) -> Any:
    """Transformée en cosinus de type II, en numpy pur.

    ``scipy.fft.dct`` ferait l'affaire, mais scipy pèse cinquante méga-octets
    pour cette seule ligne — et le cœur du projet tient à ne rien exiger de
    plus que ce que la chaîne audio apporte déjà.
    """
    n = matrice.shape[1]
    k = np.arange(n)
    base = np.cos(np.pi / n * (k[:, None] + 0.5) * k[None, :])
    return matrice @ base


def _en_mel(hertz: float) -> float:
    return 2595.0 * math.log10(1.0 + hertz / 700.0)


def _en_hertz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

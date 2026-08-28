"""VAD par énergie, sans aucune dépendance.

C'est le filet : il fonctionne partout, tout de suite, y compris sur une
machine sans onnxruntime. Il vaut moins que Silero dès qu'il y a du bruit
stationnaire — d'où le plancher de bruit adaptatif, qui suit lentement
l'ambiance et compare le niveau courant à ce plancher plutôt qu'à un seuil
absolu.

Il sert aussi de doublure rapide dans les tests, où faire tourner un réseau
de neurones pour vérifier une machine à états n'aurait aucun sens.
"""

from __future__ import annotations

from array import array
from typing import Any

from ...audio import SAMPLE_WIDTH
from ...interfaces.vad import VADEngine


def rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    echantillons = array("h")
    echantillons.frombytes(pcm[: len(pcm) - len(pcm) % SAMPLE_WIDTH])
    if not echantillons:
        return 0.0
    return (sum(float(v) * v for v in echantillons) / len(echantillons)) ** 0.5 / 32768.0


class EnergyVAD(VADEngine):
    name = "energie"

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 480,
        noise_floor: float = 0.004,
        margin: float = 3.0,
        adaptation: float = 0.02,
        creep: float = 0.001,
        **_ignored: Any,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_size = frame_size
        self.noise_floor_initial = noise_floor
        self.margin = margin
        self.adaptation = adaptation
        self.creep = creep
        self._plancher = noise_floor

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return self._frame_size

    def available(self) -> bool:
        return True

    def reset(self) -> None:
        self._plancher = self.noise_floor_initial

    def speech_probability(self, frame: bytes) -> float:
        niveau = rms(frame)
        seuil = self._plancher * self.margin
        if niveau <= seuil:
            # Sous le seuil : le plancher suit franchement l'ambiance.
            self._plancher += self.adaptation * (niveau - self._plancher)
            self._plancher = max(self._plancher, 1e-5)
            return 0.0

        # Au-dessus du seuil, le plancher monte quand même, mais vingt fois
        # plus lentement. Une phrase de quelques secondes n'y change rien ;
        # un ventilateur qui se met en marche finit par être absorbé, au lieu
        # d'être pris pour une parole ininterrompue.
        self._plancher += self.creep * (niveau - self._plancher)
        # Rapport au seuil, ramené dans [0, 1] : 1 dès qu'on est deux fois
        # au-dessus, ce qui correspond à une voix nette.
        return min(1.0, (niveau - seuil) / max(seuil, 1e-6))

    def describe(self) -> str:
        return f"energie(plancher={self._plancher:.4f})"

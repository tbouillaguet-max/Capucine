"""Interface de détection du mot d'éveil."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WakeEvent:
    word: str
    score: float
    timestamp: float


class WakeWordEngine(ABC):
    """Contrat minimal d'un détecteur de mot d'éveil.

    Le moteur consomme des trames audio en continu et doit rester très peu
    coûteux : sur Raspberry Pi il tourne en permanence.
    """

    name: str = "wake"

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Fréquence d'échantillonnage attendue des trames."""

    @property
    @abstractmethod
    def frame_length(self) -> int:
        """Nombre d'échantillons par trame attendue par ``process``."""

    @abstractmethod
    def available(self) -> bool:
        """Ne lève jamais."""

    @abstractmethod
    def process(self, frame: bytes) -> WakeEvent | None:
        """Consomme une trame PCM 16 bits et retourne un événement si le mot
        d'éveil est détecté."""

    def reset(self) -> None:
        """Vide l'état interne, après un déclenchement par exemple."""

    def close(self) -> None:
        ...

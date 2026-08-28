"""Interface de détection d'activité vocale.

Le VAD ne décide pas *ce qui* est dit, seulement *quand* on parle. Deux usages
dans Capucine, avec des réglages différents :

* terminer un énoncé sans couper l'utilisateur (``Endpointer``) ;
* repérer qu'on lui coupe la parole pendant qu'elle répond (barge-in), où le
  seuil doit être plus haut parce que le micro entend le haut-parleur.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class VADEngine(ABC):
    """Contrat minimal d'un détecteur d'activité vocale."""

    name: str = "vad"

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def frame_size(self) -> int:
        """Nombre d'échantillons par appel à ``speech_probability``.

        Silero exige exactement 512 échantillons à 16 kHz : c'est le
        rechunker qui garantit cette taille, pas l'appelant.
        """

    @abstractmethod
    def available(self) -> bool:
        """Ne lève jamais."""

    @abstractmethod
    def speech_probability(self, frame: bytes) -> float:
        """Probabilité de parole sur une trame PCM 16 bits mono, entre 0 et 1."""

    def reset(self) -> None:
        """Vide l'état interne entre deux énoncés."""

    def close(self) -> None:
        ...

    def describe(self) -> str:
        return self.name

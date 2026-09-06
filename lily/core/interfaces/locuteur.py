"""Interface de la signature vocale.

Une signature vocale transforme quelques secondes de parole en un vecteur où
la proximité géométrique traduit « c'est la même personne qui parle ». Ce
n'est pas ce que fait la transcription — qui cherche *ce qui* est dit — ni le
VAD — qui cherche *quand* on parle. Ici on cherche **qui**.

Un seul usage dans Lily, et il est étroit : après un éveil, décider si
l'énoncé vient bien de la personne enrôlée. Un « Lily » venu de la télévision
ou d'un invité ne doit pas déclencher un tour.

Deux mises en garde valent d'être dites ici plutôt que découvertes à l'usage :

* **Ce n'est pas de l'authentification.** Une signature vocale se trompe, et
  un enregistrement de votre voix la trompe. Elle filtre le bruit ambiant
  d'une pièce, elle ne garde pas un secret.
* **Elle est liée à son décor.** Le micro, la distance, la pièce entrent dans
  la signature autant que le timbre. Changez de micro et il faut réenrôler ;
  c'est vrai de tous les moteurs, plus encore du moteur spectral.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..audio import AudioBuffer


class SpeakerEngine(ABC):
    """Contrat minimal d'un extracteur de signature vocale."""

    name: str = "locuteur"
    # Le nom du moteur voyage avec chaque signature : deux moteurs produisent
    # des espaces incomparables, et comparer à travers rendrait des scores
    # silencieusement absurdes. Même précaution que pour les plongements.
    model: str = ""

    @property
    def dimension(self) -> int:
        """Taille du vecteur rendu. Zéro tant qu'elle n'est pas connue."""
        return 0

    @abstractmethod
    def available(self) -> bool:
        """Le moteur est-il réellement utilisable maintenant ? Ne lève jamais."""

    @abstractmethod
    def encode(self, audio: AudioBuffer) -> list[float]:
        """Signature d'un extrait parlé, normalisée pour le produit scalaire.

        Rend une liste vide si l'extrait ne contient pas assez de parole pour
        qu'une signature veuille dire quelque chose — c'est un état normal, pas
        une erreur : on ne signe pas deux dixièmes de seconde de silence.
        """

    def unavailable_reason(self) -> str:
        """Pourquoi ``available()`` a dit non, en une phrase actionnable."""
        return ""

    def close(self) -> None:
        """Libère les ressources."""

    def describe(self) -> str:
        return f"{self.name} ({self.model})" if self.model else self.name

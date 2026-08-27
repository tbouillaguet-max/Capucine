"""Interface de synthèse vocale."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..audio import AudioChunk

__all__ = ["AudioChunk", "TTSEngine"]


class TTSEngine(ABC):
    """Contrat minimal d'un moteur de synthèse vocale local.

    La synthèse est exposée en **flux de morceaux, une phrase par morceau**.
    C'est ce qui permet de commencer à parler avant la fin de la génération —
    et, à l'étape 3, de s'arrêter net entre deux phrases quand l'utilisateur
    coupe la parole.
    """

    name: str = "tts"

    @abstractmethod
    def available(self) -> bool:
        """Ne lève jamais."""

    @abstractmethod
    def synthesize(self, text: str, cancel: threading.Event | None = None) -> Iterator[AudioChunk]:
        """Produit la parole correspondant à ``text``, phrase par phrase.

        L'implémentation doit consulter ``cancel`` avant chaque phrase et
        s'arrêter dès qu'il est armé.
        """

    def warmup(self) -> None:
        ...

    def close(self) -> None:
        ...

    def describe(self) -> str:
        return self.name

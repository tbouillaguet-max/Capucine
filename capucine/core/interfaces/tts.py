"""Interface de synthèse vocale."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class AudioChunk:
    """Un morceau de parole prêt à être joué."""

    pcm: bytes           # PCM entiers signés 16 bits, mono, petit-boutiste
    sample_rate: int
    text: str = ""       # le fragment de texte correspondant, pour le journal

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate if self.sample_rate else 0.0


class TTSEngine(ABC):
    """Contrat minimal d'un moteur de synthèse vocale local.

    La synthèse est exposée en flux de morceaux, phrase par phrase : c'est ce
    qui permet de commencer à parler avant la fin de la génération, et de
    s'arrêter net au milieu quand l'utilisateur coupe la parole (barge-in).
    """

    name: str = "tts"

    @abstractmethod
    def available(self) -> bool:
        """Ne lève jamais."""

    @abstractmethod
    def synthesize(self, text: str, cancel: threading.Event | None = None) -> Iterator[AudioChunk]:
        """Produit la parole correspondant à ``text``, morceau par morceau.

        L'implémentation doit consulter ``cancel`` entre deux morceaux et
        s'arrêter dès qu'il est armé.
        """

    def warmup(self) -> None:
        ...

    def close(self) -> None:
        ...

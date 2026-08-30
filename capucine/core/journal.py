"""Le journal des compétences appelées — la mémoire courte des gestes.

Il sert à une seule chose, mais elle est jolie : permettre de dire « retiens
ça » après avoir fait quelque chose, et que Capucine en fasse une routine.
C'est de l'apprentissage par démonstration, la forme la plus honnête qui
soit — vous montrez, elle écrit.

Volontairement minuscule : une file bornée, en mémoire vive, jamais persistée.
Ce qui a été fait il y a trois jours n'a aucun intérêt ici ; ce qui vient
d'être fait en a beaucoup.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Appel:
    """Une compétence appelée, et avec quoi."""

    competence: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def decrire(self) -> str:
        if not self.arguments:
            return self.competence
        details = ", ".join(f"{cle}={valeur!r}" for cle, valeur in self.arguments.items())
        return f"{self.competence}({details})"


class JournalDesAppels:
    """Les dernières compétences exécutées avec succès, dans l'ordre."""

    def __init__(self, profondeur: int = 12) -> None:
        self._appels: deque[Appel] = deque(maxlen=profondeur)
        self._verrou = threading.Lock()

    def noter(self, competence: str, arguments: dict[str, Any] | None = None) -> None:
        with self._verrou:
            self._appels.append(Appel(competence, dict(arguments or {})))

    def recents(self, nombre: int = 3) -> list[Appel]:
        """Les ``nombre`` derniers appels, du plus ancien au plus récent."""
        with self._verrou:
            appels = list(self._appels)
        return appels[-max(0, nombre):] if nombre > 0 else []

    def vider(self) -> None:
        with self._verrou:
            self._appels.clear()

    def __len__(self) -> int:
        return len(self._appels)

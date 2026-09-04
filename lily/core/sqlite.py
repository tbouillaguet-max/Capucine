"""Les réglages communs aux trois magasins qui partagent un fichier SQLite.

`Memoire`, `Apprentissage` et `Connaissances` ouvrent chacun leur connexion sur
le même `~/.lily/memoire.sqlite`. Chaque verrou Python ne protège donc que sa
propre connexion : c'est SQLite qui arbitre entre elles, et deux réglages
décident de la façon dont il le fait.

* **`busy_timeout`.** Le fil d'indexation qui écrit des fragments pendant que
  l'apprentissage retient un mot : l'un des deux attend. Sans réglage explicite
  il attendrait le défaut du pilote — cinq secondes — dans le chemin chaud d'un
  tour de parole. Trois secondes suffisent largement à ces écritures-là, et le
  refus arrive avant que l'utilisateur ne croie à un blocage.

* **`synchronous = NORMAL`.** Sous WAL, c'est le réglage recommandé : la base
  reste cohérente après un plantage du programme comme après une coupure de
  courant ; seules les toutes dernières transactions peuvent manquer. En
  échange, on cesse de payer un `fsync` par `commit` — ce qui est l'essentiel
  du coût d'écriture sur la carte SD d'un Raspberry Pi.
"""

from __future__ import annotations

import contextlib
import sqlite3

ATTENTE_MS = 3000


def regler_la_base(connexion: sqlite3.Connection, attente_ms: int = ATTENTE_MS) -> None:
    """Applique les réglages communs. Ne lève pas : une base qui refuse un
    PRAGMA reste utilisable, seulement moins bien réglée."""
    for pragma in (
        "PRAGMA journal_mode=WAL",
        f"PRAGMA busy_timeout={int(attente_ms)}",
        "PRAGMA synchronous=NORMAL",
    ):
        # Un pilote qui refuse un PRAGMA laisse une base utilisable, seulement
        # moins bien réglée : ce n'est pas une raison de ne pas démarrer.
        with contextlib.suppress(sqlite3.Error):  # pragma: no cover - pilote exotique
            connexion.execute(pragma)

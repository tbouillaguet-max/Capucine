"""Vectoriseur de repli, sans modèle et sans dépendance.

**Il ne comprend rien.** Il projette les mots d'un texte dans un espace de
dimension fixe par hachage : deux textes qui partagent des mots se retrouvent
proches, deux textes qui disent la même chose avec d'autres mots restent
loin. C'est du lexical déguisé en vectoriel.

Il existe pour deux raisons honnêtes : les tests éprouvent toute la mécanique
d'indexation sans télécharger un modèle, et ``--llm mock`` permet de voir la
recherche fonctionner sur une machine fraîche. Pour de la vraie recherche par
le sens, installez ``nomic-embed-text`` — Lily le dit d'elle-même quand
elle tourne sur ce repli.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence

from ...interfaces.embeddings import EmbeddingEngine
from ...text import normalize


class HachageEmbeddings(EmbeddingEngine):
    name = "hachage"
    model = "hachage-512"

    def __init__(self, dimension: int = 512, **_ignored: object) -> None:
        self.dimension = int(dimension)

    def available(self) -> bool:
        return True

    def encode(self, textes: Sequence[str]) -> list[list[float]]:
        return [self._vectoriser(texte) for texte in textes]

    def _vectoriser(self, texte: str) -> list[float]:
        vecteur = [0.0] * self.dimension
        mots = Counter(normalize(texte).split())
        for mot, compte in mots.items():
            empreinte = hashlib.blake2b(mot.encode("utf-8"), digest_size=8).digest()
            case = int.from_bytes(empreinte[:4], "big") % self.dimension
            # Poids toujours positifs, et pas de signe alterné comme dans le
            # « hashing trick » classique : deux mots qui tombent dans la même
            # case s'y annuleraient exactement une fois sur deux, et le mot
            # cherché disparaîtrait du vecteur. Sans signe, une collision ne
            # peut que gonfler un peu la similarité — une erreur bénigne.
            # Fréquence amortie : un mot répété dix fois ne pèse pas dix fois.
            vecteur[case] += 1.0 + math.log(compte)
        norme = math.sqrt(sum(valeur * valeur for valeur in vecteur))
        if norme == 0.0:
            return vecteur
        return [valeur / norme for valeur in vecteur]

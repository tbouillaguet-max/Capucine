"""Interface du moteur de plongements lexicaux.

Un plongement transforme une phrase en un vecteur de nombres où la proximité
géométrique traduit la proximité de sens. C'est ce qui permet à « combien on
a perdu au premier trimestre » de retrouver un passage qui dit « la perte de
Q1 s'élève à », alors qu'ils n'ont pas un mot en commun.

Interface **séparée** de ``LLMEngine``, et ce n'est pas un détail : les
modèles de plongement ne sont pas des modèles de dialogue. Sous Ollama c'est
un autre modèle (``nomic-embed-text``, 274 Mo) ; sous llama.cpp c'est une
autre instance, ouverte avec ``embedding=True``. Les mélanger obligerait à
recharger un 7B pour vectoriser une phrase de dix mots.

Comme partout ailleurs : tout tourne en local, et un moteur absent est un
état, jamais une erreur — la recherche se rabat alors sur le lexical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingEngine(ABC):
    """Contrat minimal d'un vectoriseur local."""

    name: str = "embeddings"
    # Le nom du modèle est stocké avec chaque vecteur : deux modèles produisent
    # des espaces incomparables, et comparer à travers donnerait des résultats
    # silencieusement absurdes.
    model: str = ""

    @abstractmethod
    def available(self) -> bool:
        """Le moteur est-il réellement utilisable maintenant ? Ne lève jamais."""

    @abstractmethod
    def encode(self, textes: Sequence[str]) -> list[list[float]]:
        """Vectorise un lot de textes, dans l'ordre reçu.

        Le lot est explicite : vectoriser cent fragments en un appel coûte
        bien moins cher que cent appels.
        """

    def unavailable_reason(self) -> str:
        """Pourquoi ``available()`` a dit non, en une phrase actionnable.

        Même raison qu'ailleurs : paquet absent, service muet et modèle non
        tiré sont trois pannes distinctes, avec trois remèdes distincts.
        """
        return ""

    def close(self) -> None:
        """Libère les ressources."""

    def describe(self) -> str:
        return f"{self.name} ({self.model})" if self.model else self.name

"""Interface du moteur de langage.

Choix assumé : le cœur n'utilise **pas** l'API de *function calling* native
des backends. Il demande une **sortie JSON contrainte par schéma**
(``format=<schema>`` côté Ollama, grammaire GBNF côté llama.cpp). Deux
raisons : c'est le seul mécanisme qui se comporte identiquement sur les deux
backends, et c'est le seul qui *garantisse* du JSON valide au lieu de
l'espérer — ce qui compte quand le modèle est un 7-8B quantifié en Q4, ou un
1-3B sur Raspberry Pi.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolCall:
    """Décision d'appeler un skill, telle que produite par le routeur."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    source: str = "llm"  # « regle » quand l'étage déterministe a suffi
    confidence: float = 0.0


class LLMEngine(ABC):
    """Contrat minimal d'un moteur de langage local."""

    name: str = "llm"

    @abstractmethod
    def available(self) -> bool:
        """Le moteur est-il réellement utilisable maintenant ?

        Ne doit jamais lever : un moteur absent est un état, pas une erreur.
        """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Complète la conversation et retourne le texte brut.

        Si ``json_schema`` est fourni, la sortie **doit** être un document JSON
        conforme à ce schéma : c'est au moteur de contraindre le décodage, pas
        à l'appelant de prier.
        """

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """Complète en flux, morceau par morceau.

        Le repli par défaut rend la réponse en un seul bloc ; les moteurs qui
        savent streamer surchargent, ce qui permettra à l'étage TTS de parler
        phrase par phrase sans attendre la fin de l'inférence.
        """
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)

    def warmup(self) -> None:
        """Charge le modèle avant le premier tour, pour ne pas payer la
        latence de chargement devant l'utilisateur."""

    def close(self) -> None:
        """Libère les ressources."""

    def describe(self) -> str:
        return self.name

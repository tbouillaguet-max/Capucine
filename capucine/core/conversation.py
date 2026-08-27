"""Persona et mémoire courte de conversation."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from .interfaces.llm import Message
from .logging import get_logger

logger = get_logger("conversation")

PERSONA_DE_SECOURS = (
    "Tu es Capucine, une assistante vocale francophone. Ton posé, concis, "
    "jamais bavarde. Tes réponses sont lues à voix haute : deux phrases au "
    "maximum, pas de liste, pas de mise en forme."
)


def load_persona(path: Path | None) -> str:
    """Charge le persona depuis un fichier texte éditable.

    Absence de fichier ou fichier illisible : on retombe sur un persona de
    secours plutôt que d'empêcher Capucine de démarrer.
    """
    if path is None:
        return PERSONA_DE_SECOURS
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("Persona illisible (%s) : %s. Persona de secours utilisé.", path, exc)
        return PERSONA_DE_SECOURS
    return text or PERSONA_DE_SECOURS


class Conversation:
    """Les ``max_turns`` derniers échanges, plus le persona."""

    def __init__(self, persona: str = PERSONA_DE_SECOURS, max_turns: int = 6) -> None:
        self.persona = persona
        self.max_turns = max_turns
        self._messages: deque[Message] = deque(maxlen=max(1, max_turns) * 2)

    def system_prompt(self) -> str:
        return self.persona

    def add_user(self, text: str) -> None:
        self._messages.append(Message(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        if text:
            self._messages.append(Message(role="assistant", content=text))

    def add_tool_result(self, skill: str, text: str) -> None:
        self._messages.append(Message(role="assistant", content=f"[{skill}] {text}"))

    def history(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

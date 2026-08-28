"""Persona et mémoire courte de conversation."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

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
    """Les ``max_turns`` derniers échanges, le persona, et ce qu'elle sait.

    Trois horizons se superposent ici :

    * le **fil courant**, borné, qui tient dans le contexte du modèle ;
    * l'**historique**, écrit au fil de l'eau dans la mémoire persistante et
      donc consultable après un redémarrage ;
    * les **faits durables**, injectés dans le persona à chaque tour — c'est
      ce qui fait qu'elle se souvient de votre prénom la semaine suivante.
    """

    def __init__(
        self,
        persona: str = PERSONA_DE_SECOURS,
        max_turns: int = 6,
        memoire: Any = None,
        session_id: int | None = None,
    ) -> None:
        self.persona = persona
        self.max_turns = max_turns
        self.memoire = memoire
        self.session_id = session_id
        self._messages: deque[Message] = deque(maxlen=max(1, max_turns) * 2)

    # -- contexte -----------------------------------------------------------
    def system_prompt(self) -> str:
        if self.memoire is None:
            return self.persona
        faits = self.memoire.bloc_de_faits()
        return f"{self.persona}\n\n{faits}" if faits else self.persona

    # -- fil courant --------------------------------------------------------
    def add_user(self, text: str) -> None:
        self._messages.append(Message(role="user", content=text))
        self._archiver("user", text)

    def add_assistant(self, text: str) -> None:
        if text:
            self._messages.append(Message(role="assistant", content=text))
            self._archiver("assistant", text)

    def add_tool_result(self, skill: str, text: str) -> None:
        contenu = f"[{skill}] {text}"
        self._messages.append(Message(role="assistant", content=contenu))
        self._archiver("assistant", contenu)

    def _archiver(self, role: str, contenu: str) -> None:
        """Écrit dans la mémoire persistante. Ne fait jamais échouer un tour."""
        if self.memoire is None or self.session_id is None:
            return
        try:
            self.memoire.ajouter_message(self.session_id, role, contenu)
        except Exception:  # pragma: no cover - un disque plein ne coupe pas la parole
            logger.exception("Archivage du message impossible.")

    def history(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        """Vide le fil courant. L'historique, lui, reste sur le disque."""
        self._messages.clear()

    # -- reprise ------------------------------------------------------------
    def reprendre(self, session_id: int, limite: int | None = None) -> int:
        """Recharge une conversation passée dans le fil courant.

        Les tours repris sont **relus, pas réécrits** : on ne veut pas les
        archiver une seconde fois. La suite de la discussion, elle, continue
        de s'écrire dans la session que l'on vient de reprendre.
        """
        if self.memoire is None:
            return 0
        limite = limite or self._messages.maxlen
        anciens = self.memoire.messages(session_id, limite=limite)
        self._messages.clear()
        for extrait in anciens:
            role = "user" if extrait.role == "user" else "assistant"
            self._messages.append(Message(role=role, content=extrait.contenu))
        self.session_id = session_id
        logger.info("Conversation #%d reprise (%d messages).", session_id, len(anciens))
        return len(anciens)

    def __len__(self) -> int:
        return len(self._messages)

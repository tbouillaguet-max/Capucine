"""Moteur LLM factice : déterministe, sans réseau, sans poids.

Il a deux usages, et les deux comptent :

* les tests s'en servent pour piloter le pipeline sans télécharger un seul
  giga-octet — la boucle de plugins est prouvée sans modèle ;
* ``--llm mock`` rend Lily utilisable en mode texte sur une machine
  fraîche : l'étage déterministe du routeur suffit à appeler les plugins,
  le LLM ne sert qu'à la conversation libre.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any

from ...interfaces.llm import LLMEngine, Message

DEFAULT_REPLY = "Je fonctionne sans modèle de langage : je sais exécuter des compétences, pas discuter."


def _sample_for_schema(schema: dict[str, Any], utterance: str = "") -> Any:
    """Valeur minimale conforme à un schéma, pour rester honnête sur le
    contrat « la sortie contrainte est toujours valide ».

    Un argument texte obligatoire reçoit la phrase de l'utilisateur : c'est ce
    qui rend ``--llm mock`` utilisable pour de vrai sur un skill comme
    ``repete``, sans prétendre comprendre le français.
    """
    kind = schema.get("type", "string")
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if schema.get("nullable"):
        return None
    if kind == "object":
        return {
            key: _sample_for_schema(value, utterance)
            for key, value in schema.get("properties", {}).items()
            if key in schema.get("required", [])
        }
    if kind == "array":
        return []
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    return schema.get("default", utterance)


class MockLLM(LLMEngine):
    name = "mock"

    def __init__(
        self,
        responses: Iterable[str] | None = None,
        default: str = DEFAULT_REPLY,
        **_ignored: Any,
    ) -> None:
        self.responses: deque[str] = deque(responses or ())
        self.default = default
        self.calls: list[dict[str, Any]] = []

    def push(self, *responses: str) -> MockLLM:
        """Programme les prochaines réponses (utilisé par les tests)."""
        self.responses.extend(responses)
        return self

    def available(self) -> bool:
        return True

    def chat(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        self.calls.append({
            "messages": [m.as_dict() for m in messages],
            "json_schema": json_schema,
        })
        if self.responses:
            return self.responses.popleft()
        if json_schema is not None:
            last_user = next(
                (m.content for m in reversed(messages) if m.role == "user"), ""
            )
            return json.dumps(_sample_for_schema(json_schema, last_user), ensure_ascii=False)
        return self.default

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        for word in self.chat(messages, **kwargs).split(" "):
            yield word + " "

"""Choix de l'outil : trois étages, du moins cher au plus faible.

Un modèle 7-8B quantifié en Q4, en français, avec une vingtaine d'outils dans
son contexte, hallucine des noms d'outils et se trompe de type d'argument
assez souvent pour casser une démonstration. Sur Raspberry Pi avec un 1-3B,
c'est pire. D'où ce routage :

1. **Étage déterministe.** Score entre la phrase entendue et les ``examples``
   du décorateur. Latence nulle, aucun modèle sollicité. C'est cet étage qui
   rend « lance un dé à vingt faces » fiable plutôt que probable.
2. **Étage LLM, en deux passes contraintes.** D'abord le *nom* de l'outil,
   contraint par une énumération — un nom halluciné devient structurellement
   impossible. Puis les *arguments*, contraints par le schéma réel de l'outil
   choisi. Deux petites générations garanties valides valent mieux qu'une
   grande espérée valide.
3. **Étage conversationnel.** Aucun outil ne convient : le modèle répond.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .interfaces.llm import LLMEngine, Message, ToolCall
from .logging import get_logger
from .plugin import SkillSpec
from .text import PhrasePreparee, extract_numbers, similarity_preparee

logger = get_logger("routeur")

NO_TOOL = "aucun"

# Poids par nature de phrase : un exemple fourni par l'auteur du plugin est un
# signal bien plus sûr que sa description.
_WEIGHTS = {"example": 1.0, "name": 0.85, "description": 0.45}


@dataclass
class Candidate:
    name: str
    score: float
    matched: str = ""


@dataclass
class RouteDecision:
    """Résultat du routage : soit un appel d'outil, soit une réponse."""

    tool_call: ToolCall | None = None
    answer: str | None = None
    tier: str = "conversation"
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def has_tool(self) -> bool:
        return self.tool_call is not None


class Router:
    def __init__(
        self,
        llm: LLMEngine,
        *,
        direct_threshold: float = 0.72,
        shortlist_threshold: float = 0.35,
        shortlist_size: int = 5,
        allow_number_extraction: bool = True,
        temperature: float = 0.0,
        apprentissage: Any = None,
    ) -> None:
        self.llm = llm
        # Ce que Lily a retenu de vos formulations. Consulté à chaque tour,
        # servi depuis un cache : aucun accès disque dans le chemin chaud.
        self.apprentissage = apprentissage
        self.direct_threshold = direct_threshold
        self.shortlist_threshold = shortlist_threshold
        self.shortlist_size = shortlist_size
        self.allow_number_extraction = allow_number_extraction
        self.temperature = temperature

    # -- étage 0 : déterministe --------------------------------------------
    def score_skills(self, utterance: str, skills: Mapping[str, SkillSpec]) -> list[Candidate]:
        apprises = (
            self.apprentissage.phrases_par_outil() if self.apprentissage is not None else {}
        )
        # Une seule normalisation de la phrase entendue, au lieu d'une par
        # comparaison : sur soixante-huit compétences, la boucle ci-dessous en
        # faisait deux cent cinquante fois le même travail.
        demande = PhrasePreparee.de(utterance)
        candidates: list[Candidate] = []
        for name, spec in skills.items():
            if spec.quarantined:
                continue
            best = 0.0
            matched = ""
            # Exemples, formulations retenues, nom lisible, description :
            # tout est préparé d'avance, ici on ne fait que comparer. Vos
            # formulations pèsent presque autant qu'un exemple d'auteur,
            # jamais davantage.
            retenues = [
                (retenue.preparee, retenue.poids, f"appris: {retenue.phrase}")
                for retenue in apprises.get(name, ())
            ]
            for phrase, poids, libelle in spec.phrases_ponderees(_WEIGHTS, retenues):
                value = similarity_preparee(demande, phrase, best / poids) * poids
                if value > best:
                    best, matched = value, libelle
            candidates.append(Candidate(name=name, score=round(best, 3), matched=matched))
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _resolve_arguments_locally(self, utterance: str, spec: SkillSpec) -> dict[str, Any] | None:
        """Complète les arguments sans LLM, quand c'est sans ambiguïté.

        Règle volontairement étroite : uniquement si tous les paramètres ont une
        valeur par défaut, et — pour un argument numérique — s'il n'y a
        qu'un seul paramètre numérique et qu'un seul nombre dans la phrase.
        Au moindre doute, on laisse le modèle décider.
        """
        if spec.required_parameters:
            return None
        properties: dict[str, Any] = spec.parameters_schema.get("properties", {})
        numeric = [
            key for key, schema in properties.items()
            if schema.get("type") in ("integer", "number") and "enum" not in schema
        ]
        if not numeric:
            return {}
        if not self.allow_number_extraction or len(numeric) != 1:
            return None
        numbers = extract_numbers(utterance)
        if len(numbers) != 1:
            return {} if not numbers else None
        return {numeric[0]: numbers[0]}

    # -- étage 1 : LLM contraint -------------------------------------------
    def _select_tool_with_llm(
        self, utterance: str, shortlist: list[SkillSpec], history: list[Message], system: str
    ) -> str:
        names = [spec.name for spec in shortlist]
        catalogue = "\n".join(
            f"- {spec.name} : {spec.tool_schema['function']['description'].splitlines()[0]}"
            for spec in shortlist
        )
        instructions = (
            f"{system}\n\n"
            "Tu choisis l'outil à utiliser pour répondre à la demande de l'utilisateur.\n"
            f"Outils disponibles :\n{catalogue}\n\n"
            f"Réponds en JSON avec la clé « outil ». Si aucun outil ne convient, "
            f"réponds « {NO_TOOL} »."
        )
        schema = {
            "type": "object",
            "properties": {"outil": {"type": "string", "enum": [*names, NO_TOOL]}},
            "required": ["outil"],
        }
        messages = [
            Message(role="system", content=instructions),
            *history,
            Message(role="user", content=utterance),
        ]
        raw = self.llm.chat(messages, json_schema=schema, temperature=self.temperature, max_tokens=64)
        payload = _load_json(raw)
        chosen = str(payload.get("outil", NO_TOOL)) if payload else NO_TOOL
        return chosen if chosen in names else NO_TOOL

    def _fill_arguments_with_llm(
        self, utterance: str, spec: SkillSpec, history: list[Message]
    ) -> dict[str, Any]:
        if not spec.parameter_names:
            return {}
        description = spec.tool_schema["function"]["description"]
        instructions = (
            f"Tu extrais les arguments de l'outil « {spec.name} » depuis la demande.\n"
            f"Rôle de l'outil : {description}\n"
            "Réponds uniquement en JSON. N'invente pas de valeur : si un argument "
            "n'est pas dit explicitement, omets-le pour garder sa valeur par défaut."
        )
        # Le schéma des paramètres pilote directement le décodage contraint :
        # les types sont donc garantis, pas seulement suggérés.
        schema = dict(spec.parameters_schema)
        messages = [
            Message(role="system", content=instructions),
            *history,
            Message(role="user", content=utterance),
        ]
        raw = self.llm.chat(messages, json_schema=schema, temperature=self.temperature, max_tokens=256)
        return _load_json(raw) or {}

    # -- étage 2 : conversation --------------------------------------------
    def answer_freely(self, utterance: str, history: list[Message], system: str) -> str:
        messages = [
            Message(role="system", content=system),
            *history,
            Message(role="user", content=utterance),
        ]
        return self.llm.chat(messages, temperature=0.6, max_tokens=200).strip()

    def stream_answer(self, utterance: str, history: list[Message], system: str) -> Iterator[str]:
        """Même réponse, mais en flux : la synthèse peut commencer à parler
        dès la première phrase terminée, sans attendre la fin de l'inférence."""
        messages = [
            Message(role="system", content=system),
            *history,
            Message(role="user", content=utterance),
        ]
        yield from self.llm.stream(messages, temperature=0.6, max_tokens=200)

    # -- orchestration ------------------------------------------------------
    def route(
        self,
        utterance: str,
        skills: Mapping[str, SkillSpec],
        history: list[Message] | None = None,
        system: str = "",
    ) -> RouteDecision:
        history = history or []
        candidates = self.score_skills(utterance, skills)

        if not candidates:
            return RouteDecision(answer=None, tier="conversation", candidates=[])

        best = candidates[0]
        if best.score >= self.direct_threshold:
            spec = skills[best.name]
            arguments = self._resolve_arguments_locally(utterance, spec)
            if arguments is not None:
                logger.debug("Étage déterministe : %s (score %.2f)", best.name, best.score)
                return RouteDecision(
                    tool_call=ToolCall(
                        name=best.name, arguments=arguments, source="regle", confidence=best.score
                    ),
                    tier="regle",
                    candidates=candidates[: self.shortlist_size],
                )

        shortlist = [
            skills[c.name] for c in candidates[: self.shortlist_size]
            if c.score >= self.shortlist_threshold
        ]
        if not shortlist:
            # Rien ne ressemble à une compétence : on garde quand même les
            # mieux classées, le modèle voit parfois ce que le score rate.
            shortlist = [skills[c.name] for c in candidates[: self.shortlist_size]]

        chosen = self._select_tool_with_llm(utterance, shortlist, history, system)
        if chosen == NO_TOOL:
            return RouteDecision(tier="conversation", candidates=candidates[: self.shortlist_size])

        spec = skills[chosen]
        arguments = self._fill_arguments_with_llm(utterance, spec, history)
        confidence = next((c.score for c in candidates if c.name == chosen), 0.0)
        return RouteDecision(
            tool_call=ToolCall(name=chosen, arguments=arguments, source="llm", confidence=confidence),
            tier="llm",
            candidates=candidates[: self.shortlist_size],
        )


def _load_json(raw: str) -> dict[str, Any] | None:
    """Lit le JSON du modèle, en tolérant un peu de bavardage autour.

    Avec un décodage contraint, ce filet ne sert jamais. Il existe pour les
    moteurs qui n'appliquent pas la contrainte.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            logger.debug("Réponse non JSON du moteur : %r", raw[:200])
            return None
        try:
            value = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            logger.debug("JSON illisible dans la réponse : %r", raw[:200])
            return None
    return value if isinstance(value, dict) else None

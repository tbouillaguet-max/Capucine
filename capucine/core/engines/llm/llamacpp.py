"""Moteur LLM adossé à ``llama-cpp-python`` (llama.cpp en processus).

Alternative à Ollama, activable par une ligne de configuration. Elle évite le
démon, au prix d'une installation plus délicate (roues CUDA sous Windows) et
du fait que le modèle vit dans le processus de Capucine.

La sortie structurée passe par ``response_format={"type": "json_object",
"schema": …}``, que llama.cpp traduit en grammaire GBNF et applique à
l'échantillonnage : le JSON est structurellement garanti, pas espéré.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ...errors import EngineUnavailable
from ...interfaces.llm import LLMEngine, Message
from ...logging import get_logger

logger = get_logger("llm.llamacpp")


class LlamaCppLLM(LLMEngine):
    name = "llamacpp"

    def __init__(
        self,
        model_path: str | Path = "",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        chat_format: str | None = None,
        temperature: float = 0.3,
        seed: int | None = None,
        verbose: bool = False,
        **_ignored: Any,
    ) -> None:
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.chat_format = chat_format
        self.temperature = temperature
        self.seed = seed
        self.verbose = verbose
        self._llama: Any = None

    def _get_llama(self) -> Any:
        if self._llama is None:
            if not self.model_path or not self.model_path.exists():
                raise EngineUnavailable(
                    f"Modèle GGUF introuvable : {self.model_path}. "
                    "Renseignez llm.model_path dans la configuration."
                )
            try:
                from llama_cpp import Llama
            except ImportError as exc:  # pragma: no cover - dépend de l'install
                raise EngineUnavailable(
                    "Le paquet « llama-cpp-python » est absent. "
                    "Installez-le avec : pip install llama-cpp-python"
                ) from exc
            kwargs: dict[str, Any] = {
                "model_path": str(self.model_path),
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "verbose": self.verbose,
            }
            if self.n_threads:
                kwargs["n_threads"] = self.n_threads
            if self.chat_format:
                kwargs["chat_format"] = self.chat_format
            if self.seed is not None:
                kwargs["seed"] = self.seed
            logger.info("Chargement du modèle %s", self.model_path.name)
            self._llama = Llama(**kwargs)
        return self._llama

    def available(self) -> bool:
        if not self.model_path or not self.model_path.exists():
            return False
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return False
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
        llama = self._get_llama()
        kwargs: dict[str, Any] = {
            "messages": [m.as_dict() for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        if json_schema is not None:
            # Contrainte par grammaire GBNF dérivée du schéma.
            kwargs["response_format"] = {"type": "json_object", "schema": json_schema}
        response = llama.create_chat_completion(**kwargs)
        return response["choices"][0]["message"].get("content") or ""

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        llama = self._get_llama()
        kwargs: dict[str, Any] = {
            "messages": [m.as_dict() for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "stream": True,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        for chunk in llama.create_chat_completion(**kwargs):
            delta = chunk["choices"][0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                yield piece

    def warmup(self) -> None:
        try:
            self._get_llama()
        except Exception as exc:  # pragma: no cover
            logger.warning("Préchauffage llama.cpp impossible : %s", exc)

    def close(self) -> None:
        self._llama = None

    def describe(self) -> str:
        return f"llamacpp:{self.model_path.name if self.model_path else '?'}"

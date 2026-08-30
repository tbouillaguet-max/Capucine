"""Plongements via ``llama-cpp-python``, en processus.

Instance **distincte** de celle du dialogue : llama.cpp veut ``embedding=True``
et une mise en commun (*pooling*) à la construction, et un modèle de plongement
n'est de toute façon pas un modèle de dialogue. Pointez ``model_path`` sur un
GGUF d'encodeur — par exemple ``nomic-embed-text-v1.5.Q4_K_M.gguf``.

``Llama.embed`` sait normaliser lui-même (``normalize=True``) ; on le lui
demande, ce qui rend la similarité cosinus égale à un simple produit scalaire.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...errors import EngineUnavailable
from ...interfaces.embeddings import EmbeddingEngine
from ...logging import get_logger

logger = get_logger("embeddings.llamacpp")


class LlamaCppEmbeddings(EmbeddingEngine):
    name = "llamacpp"

    def __init__(
        self,
        model_path: str | Path = "",
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        n_batch: int = 512,
        verbose: bool = False,
        **_ignored: Any,
    ) -> None:
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.model = self.model_path.name if self.model_path else ""
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.verbose = verbose
        self._llama: Any = None
        self._raison = ""

    def _get_llama(self) -> Any:
        if self._llama is None:
            if not self.model_path or not self.model_path.exists():
                raise EngineUnavailable(
                    f"Modèle de plongement introuvable : {self.model_path}. "
                    "Renseignez connaissances.model_path dans la configuration."
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
                "n_batch": self.n_batch,
                "n_gpu_layers": self.n_gpu_layers,
                "embedding": True,
                "verbose": self.verbose,
            }
            if self.n_threads:
                kwargs["n_threads"] = self.n_threads
            logger.info("Chargement du modèle de plongement %s", self.model_path.name)
            self._llama = Llama(**kwargs)
        return self._llama

    def available(self) -> bool:
        if not self.model_path or not self.model_path.exists():
            self._raison = (
                f"Modèle GGUF introuvable : {self.model_path or '(non renseigné)'}. "
                "Renseignez connaissances.model_path dans la configuration."
            )
            return False
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            self._raison = (
                "Le paquet « llama-cpp-python » est absent. Installez-le avec : "
                "pip install llama-cpp-python"
            )
            return False
        self._raison = ""
        return True

    def unavailable_reason(self) -> str:
        return self._raison

    def encode(self, textes: Sequence[str]) -> list[list[float]]:
        if not textes:
            return []
        # `embed` rend une liste de vecteurs pour une liste, un seul vecteur
        # pour une chaîne : on ne lui passe jamais de chaîne, pour n'avoir
        # qu'une forme de retour à traiter.
        vecteurs = self._get_llama().embed(list(textes), normalize=True)
        return [list(map(float, vecteur)) for vecteur in vecteurs]

    def close(self) -> None:
        self._llama = None

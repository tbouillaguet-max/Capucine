"""Plongements via Ollama, en local uniquement.

Même garde-fou que le moteur de dialogue : l'hôte est validé et doit rester
une adresse de bouclage. Vectoriser un document, c'est envoyer son contenu au
moteur — la dernière chose qu'on veut est que cet envoi sorte de la machine.

Modèle par défaut : ``nomic-embed-text`` (274 Mo, 768 dimensions, multilingue,
français inclus). ``ollama pull nomic-embed-text`` et c'est tout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...errors import EngineUnavailable
from ...interfaces.embeddings import EmbeddingEngine
from ...logging import get_logger
from ..llm.ollama import _service_muet, exiger_hote_local

logger = get_logger("embeddings.ollama")


class OllamaEmbeddings(EmbeddingEngine):
    name = "ollama"

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://127.0.0.1:11434",
        keep_alive: str = "5m",
        timeout: float = 120.0,
        **_ignored: Any,
    ) -> None:
        exiger_hote_local(host)
        self.model = model
        self.host = host
        self.keep_alive = keep_alive
        self.timeout = timeout
        self._client: Any = None
        self._raison = ""

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:  # pragma: no cover - dépend de l'install
                raise EngineUnavailable(
                    "Le paquet « ollama » est absent. Installez-le avec : pip install ollama"
                ) from exc
            self._client = ollama.Client(host=self.host, timeout=self.timeout)
        return self._client

    def available(self) -> bool:
        """Le service répond-il, et le modèle de plongement est-il tiré ?

        On vérifie le modèle et pas seulement le service : un ``ollama list``
        qui répond alors que ``nomic-embed-text`` n'est pas tiré donnerait une
        erreur au premier document indexé, c'est-à-dire au pire moment.
        """
        try:
            listing = self._get_client().list()
        except EngineUnavailable as exc:
            self._raison = str(exc)
            return False
        except Exception as exc:
            self._raison = _service_muet(self.host)
            logger.debug("Ollama injoignable pour les plongements : %s", exc)
            return False
        # `list()` rend un objet pydantic dont `.models` porte des entrées à
        # champ `model`. On accepte aussi la forme dictionnaire : le client a
        # changé de représentation entre deux versions, pas de raison de casser.
        entrees = getattr(listing, "models", None)
        if entrees is None and isinstance(listing, dict):
            entrees = listing.get("models", [])
        noms: list[str] = []
        for entree in entrees or []:
            nom = getattr(entree, "model", None)
            if nom is None and isinstance(entree, dict):
                nom = entree.get("model") or entree.get("name")
            if nom:
                noms.append(str(nom))
        # Ollama nomme « nomic-embed-text:latest » ce qu'on demande sous
        # « nomic-embed-text » : on compare sur la racine.
        racine = self.model.split(":")[0]
        if not any(nom.split(":")[0] == racine for nom in noms):
            self._raison = (
                f"Le service répond, mais le modèle de plongement « {self.model} » "
                f"n'est pas tiré. Faites : ollama pull {racine}"
            )
            return False
        self._raison = ""
        return True

    def unavailable_reason(self) -> str:
        return self._raison

    def encode(self, textes: Sequence[str]) -> list[list[float]]:
        if not textes:
            return []
        reponse = self._get_client().embed(
            model=self.model, input=list(textes), keep_alive=self.keep_alive
        )
        vecteurs = [list(map(float, vecteur)) for vecteur in reponse.embeddings]
        if len(vecteurs) != len(textes):
            raise EngineUnavailable(
                f"Le moteur a rendu {len(vecteurs)} vecteurs pour {len(textes)} textes."
            )
        return vecteurs

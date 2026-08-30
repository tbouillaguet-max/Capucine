"""Moteur LLM adossé à Ollama, en local uniquement.

Ollama tourne comme un service sur la machine. Ce n'est pas un service tiers :
l'hôte est validé et doit rester une adresse de bouclage, sans quoi le moteur
refuse de démarrer. Aucune requête ne sort de la machine, Wi-Fi coupé compris.

Trois raisons d'en faire le défaut sur PC plutôt que ``llama-cpp-python`` :
aucune compilation à l'installation, le modèle vit dans un **processus
séparé** — un crash du moteur ne tue pas Capucine —, et la gestion mémoire du
chargement/déchargement est faite pour nous.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

from ...errors import EngineUnavailable
from ...interfaces.llm import LLMEngine, Message
from ...logging import get_logger

logger = get_logger("llm.ollama")

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0", ""}


def exiger_hote_local(host: str) -> None:
    """Refuse tout hôte qui n'est pas cette machine.

    Public parce que le moteur de plongements applique exactement la même
    règle : vectoriser un document, c'est en envoyer le contenu au moteur.
    """
    parsed = urlparse(host if "//" in host else f"http://{host}")
    hostname = (parsed.hostname or "").lower()
    if hostname not in _LOOPBACK_HOSTS:
        raise EngineUnavailable(
            f"Capucine refuse de contacter un hôte non local : {host!r}. "
            "Le moteur Ollama doit tourner sur cette machine."
        )


def _service_muet(host: str) -> str:
    """Le message quand le service ne répond pas — partagé avec les plongements."""
    return (
        f"Le service Ollama ne répond pas sur {host}. Sous Windows il démarre "
        "avec la session (icône dans la zone de notification) ; sous Linux, "
        "« systemctl status ollama », ou « ollama serve » dans un autre terminal."
    )


class OllamaLLM(LLMEngine):
    name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct-q4_K_M",
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.3,
        num_ctx: int = 4096,
        keep_alive: str = "10m",
        timeout: float = 60.0,
        **_ignored: Any,
    ) -> None:
        exiger_hote_local(host)
        self.model = model
        self.host = host
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.timeout = timeout
        self._client: Any = None
        self._raison = ""

    # -- accès paresseux au client -----------------------------------------
    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import ollama  # import paresseux : --text sans LLM ne le paie pas
            except ImportError as exc:  # pragma: no cover - dépend de l'install
                raise EngineUnavailable(
                    "Le paquet « ollama » est absent. Installez-le avec : pip install ollama"
                ) from exc
            self._client = ollama.Client(host=self.host, timeout=self.timeout)
        return self._client

    def available(self) -> bool:
        try:
            self._get_client().list()
        except EngineUnavailable as exc:
            # Le paquet Python manque : rien à voir avec le service, qui peut
            # très bien tourner — « ollama pull » marche par le binaire natif,
            # pas par ce paquet-là. Confondre les deux envoie chercher au
            # mauvais endroit.
            self._raison = str(exc)
            return False
        except Exception as exc:
            self._raison = _service_muet(self.host)
            logger.debug("Ollama injoignable : %s", exc)
            return False
        self._raison = ""
        return True

    def unavailable_reason(self) -> str:
        return self._raison

    def _options(self, temperature: float | None, max_tokens: int | None, stop: list[str] | None) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "num_ctx": self.num_ctx,
        }
        if max_tokens:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = stop
        return options

    def chat(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        client = self._get_client()
        response = client.chat(
            model=self.model,
            messages=[m.as_dict() for m in messages],
            format=json_schema if json_schema is not None else None,
            options=self._options(temperature, max_tokens, stop),
            keep_alive=self.keep_alive,
        )
        return _content_of(response)

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        client = self._get_client()
        for part in client.chat(
            model=self.model,
            messages=[m.as_dict() for m in messages],
            options=self._options(temperature, max_tokens, stop),
            keep_alive=self.keep_alive,
            stream=True,
        ):
            piece = _content_of(part)
            if piece:
                yield piece

    def warmup(self) -> None:
        try:
            self.chat([Message(role="user", content="bonjour")], max_tokens=1)
        except Exception as exc:  # pragma: no cover - dépend du service
            logger.warning("Préchauffage Ollama impossible : %s", exc)

    def describe(self) -> str:
        return f"ollama:{self.model}"


def _content_of(response: Any) -> str:
    """Le client Ollama 0.6 rend un objet typé ; on accepte aussi un dict."""
    message = getattr(response, "message", None)
    if message is not None:
        return getattr(message, "content", "") or ""
    if isinstance(response, dict):
        return (response.get("message") or {}).get("content", "") or ""
    return ""

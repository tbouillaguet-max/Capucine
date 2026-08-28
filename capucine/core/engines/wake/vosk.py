"""Mot d'éveil par Vosk, à grammaire restreinte.

C'est le repli tant que le modèle openWakeWord n'est pas entraîné — et
vraisemblablement le chemin principal pendant un moment, parce qu'entraîner un
modèle personnalisé en français demande un corpus de synthèse de qualité.

Le principe : on donne au décodeur une grammaire de deux entrées seulement,
« capucine » et ``[unk]``. Il ne peut littéralement rien reconnaître d'autre,
ce qui le rend rapide et peu gourmand — il tourne en continu sur un Pi. On
lit les résultats partiels, sans attendre la fin d'un énoncé.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ...errors import EngineUnavailable
from ...interfaces.wake import WakeEvent, WakeWordEngine
from ...logging import get_logger
from ...text import normalize

logger = get_logger("wake.vosk")


class VoskWakeWord(WakeWordEngine):
    name = "vosk"

    def __init__(
        self,
        word: str = "capucine",
        variants: list[str] | None = None,
        model_path: str | Path = "models/vosk/vosk-model-small-fr-0.22",
        sample_rate: int = 16000,
        frame_size: int = 2000,
        debounce_s: float = 2.0,
        log_level: int = -1,
        **_ignored: Any,
    ) -> None:
        self.word = word
        # Les variantes élargissent la grammaire, pas la correspondance : un
        # décodeur qui a le droit de dire « capucin » se trompera moins qu'un
        # décodeur forcé de choisir entre « capucine » et rien.
        self.variants = variants or [word, "capucin", "ma capucine"]
        self.model_path = Path(model_path).expanduser()
        self._sample_rate = sample_rate
        self._frame_size = frame_size
        self.debounce_s = debounce_s
        self.log_level = log_level
        self._model: Any = None
        self._recogniser: Any = None
        self._dernier_declenchement = 0.0
        self._cible = normalize(word)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return self._frame_size

    def available(self) -> bool:
        try:
            import vosk  # noqa: F401
        except ImportError:
            return False
        return self.model_path.is_dir()

    def grammar(self) -> str:
        return json.dumps([*dict.fromkeys(self.variants), "[unk]"], ensure_ascii=False)

    def _get_recogniser(self) -> Any:
        if self._recogniser is not None:
            return self._recogniser
        try:
            import vosk
        except ImportError as exc:  # pragma: no cover
            raise EngineUnavailable(
                "Le paquet « vosk » est absent. Installez-le avec : pip install vosk"
            ) from exc
        if not self.model_path.is_dir():
            raise EngineUnavailable(
                f"Modèle Vosk introuvable : {self.model_path}. Téléchargez-le avec :\n"
                "  python -m capucine.core.downloads vosk"
            )
        if self._model is None:
            vosk.SetLogLevel(self.log_level)
            logger.info("Chargement du modèle d'éveil Vosk « %s »", self.model_path.name)
            self._model = vosk.Model(str(self.model_path))
        self._recogniser = vosk.KaldiRecognizer(self._model, self._sample_rate, self.grammar())
        return self._recogniser

    def _contient_le_mot(self, texte: str) -> bool:
        normalise = normalize(texte)
        return bool(normalise) and self._cible in normalise

    def process(self, frame: bytes) -> WakeEvent | None:
        maintenant = time.monotonic()
        if maintenant - self._dernier_declenchement < self.debounce_s:
            return None

        recogniser = self._get_recogniser()
        if recogniser.AcceptWaveform(frame):
            texte = (json.loads(recogniser.Result()) or {}).get("text", "")
        else:
            texte = (json.loads(recogniser.PartialResult()) or {}).get("partial", "")

        if not self._contient_le_mot(texte):
            return None
        self._dernier_declenchement = maintenant
        self.reset()
        logger.info("Mot d'éveil détecté (vosk) : %r", texte)
        return WakeEvent(word=self.word, score=1.0, timestamp=maintenant)

    def reset(self) -> None:
        # On repart d'un décodeur neuf : conserver l'état ferait re-déclencher
        # sur le même mot aux trames suivantes.
        self._recogniser = None

    def close(self) -> None:
        self._recogniser = None
        self._model = None

    def describe(self) -> str:
        return f"vosk:{self.word}"

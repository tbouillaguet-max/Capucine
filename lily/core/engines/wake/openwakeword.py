"""Mot d'éveil par openWakeWord.

Le modèle « lily » n'existe pas dans les modèles pré-entraînés : il faut
l'entraîner, ce que fait ``tools/entrainer_lily.py``. Tant qu'il n'est pas
prêt, le repli Vosk à grammaire restreinte prend le relais — c'est un état
normal du projet, pas une panne, et le message d'erreur le dit.

Deux choix par rapport aux défauts de la bibliothèque :

* ``inference_framework="onnx"`` plutôt que ``tflite``. ``tflite-runtime`` n'a
  pas de roue pour toutes les combinaisons Windows/Python 3.11+, alors
  qu'``onnxruntime`` s'installe partout — et sert déjà au VAD.
* un **anti-rebond** après déclenchement : sans lui, une détection produit une
  rafale d'événements sur les trames suivantes, puisque le mot est encore dans
  la fenêtre d'analyse.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ...errors import EngineUnavailable
from ...interfaces.wake import WakeEvent, WakeWordEngine
from ...logging import get_logger

logger = get_logger("wake.oww")

# openWakeWord raisonne par tranches de 1280 échantillons (80 ms) à 16 kHz.
TAILLE_DE_TRAME = 1280
EXTENSIONS = (".onnx", ".tflite")


class OpenWakeWordEngine(WakeWordEngine):
    name = "openwakeword"

    def __init__(
        self,
        word: str = "lily",
        models_dir: str | Path = "models/wake",
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        inference_framework: str = "onnx",
        vad_threshold: float = 0.0,
        debounce_s: float = 2.0,
        enable_speex_noise_suppression: bool = False,
        **_ignored: Any,
    ) -> None:
        self.word = word
        self.models_dir = Path(models_dir).expanduser()
        self.explicit_path = Path(model_path).expanduser() if model_path else None
        self.threshold = threshold
        self.inference_framework = inference_framework
        self.vad_threshold = vad_threshold
        self.debounce_s = debounce_s
        self.enable_speex_noise_suppression = enable_speex_noise_suppression
        self._model: Any = None
        self._dernier_declenchement = 0.0

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_size(self) -> int:
        return TAILLE_DE_TRAME

    def model_path(self) -> Path | None:
        if self.explicit_path is not None:
            return self.explicit_path if self.explicit_path.exists() else None
        for extension in EXTENSIONS:
            candidat = self.models_dir / f"{self.word}{extension}"
            if candidat.exists():
                return candidat
        return None

    def available(self) -> bool:
        try:
            import openwakeword  # noqa: F401
        except ImportError:
            return False
        return self.model_path() is not None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from openwakeword.model import Model
        except ImportError as exc:  # pragma: no cover - dépend de l'install
            raise EngineUnavailable(
                "Le paquet « openwakeword » est absent. "
                "Installez-le avec : pip install openwakeword onnxruntime"
            ) from exc

        chemin = self.model_path()
        if chemin is None:
            raise EngineUnavailable(
                f"Modèle de mot d'éveil « {self.word} » introuvable dans "
                f"{self.models_dir}. Il n'existe pas de modèle pré-entraîné pour "
                "« lily » : entraînez-le avec\n"
                "  python tools/entrainer_lily.py --preparer\n"
                "ou, en attendant, utilisez le repli Vosk : wake.engine = \"vosk\"."
            )
        logger.info("Chargement du modèle d'éveil %s", chemin.name)
        self._model = Model(
            wakeword_models=[str(chemin)],
            inference_framework=self.inference_framework,
            vad_threshold=self.vad_threshold,
            enable_speex_noise_suppression=self.enable_speex_noise_suppression,
        )
        return self._model

    def process(self, frame: bytes) -> WakeEvent | None:
        import numpy as np

        maintenant = time.monotonic()
        if maintenant - self._dernier_declenchement < self.debounce_s:
            # Anti-rebond : le mot est encore dans la fenêtre d'analyse.
            return None

        scores = self._get_model().predict(np.frombuffer(frame, dtype=np.int16))
        if not scores:
            return None
        nom, score = max(scores.items(), key=lambda paire: paire[1])
        if score < self.threshold:
            return None
        self._dernier_declenchement = maintenant
        self.reset()
        logger.info("Mot d'éveil détecté : %s (%.2f)", nom, score)
        return WakeEvent(word=nom, score=float(score), timestamp=maintenant)

    def score(self, frame: bytes) -> float:
        """Le score brut de la trame, sans seuil ni anti-rebond.

        Sert à mesurer un seuil sur un corpus étiqueté plutôt qu'à le
        deviner : ``tools/entrainer_lily.py seuil``.
        """
        import numpy as np

        scores = self._get_model().predict(np.frombuffer(frame, dtype=np.int16))
        return float(max(scores.values())) if scores else 0.0

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()

    def close(self) -> None:
        self._model = None

    def describe(self) -> str:
        return f"openwakeword:{self.word}"

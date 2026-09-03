"""Silero VAD, exécuté par onnxruntime — sans torch.

Divergence assumée, et elle compte pour le Raspberry Pi. Le paquet
``silero-vad`` importe ``torch`` et ``torchaudio`` dès son ``__init__``, y
compris sur le chemin ONNX : l'installer coûte plusieurs centaines de
méga-octets sur un Pi, pour un modèle de moins d'un méga-octet.

Or le fichier ``silero_vad.onnx`` est **livré dans le paquet**. On le charge
donc directement avec ``onnxruntime`` : mêmes poids, même modèle, sans la
chaîne torch. ``pip install --no-deps silero-vad`` suffit à disposer du
fichier ; on accepte aussi un chemin explicite ou ``models/silero/``.

Le graphe attend trois entrées : ``input`` (float32, 512 échantillons à
16 kHz), ``state`` (float32 [2, 1, 128]) et ``sr`` (int64). Il rend la
probabilité de parole et le nouvel état, qu'on reporte d'une trame à l'autre.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from ...errors import EngineUnavailable
from ...interfaces.vad import VADEngine
from ...logging import get_logger

logger = get_logger("vad.silero")

# Silero v5 impose ces tailles ; ce ne sont pas des réglages.
TAILLES_DE_TRAME = {16000: 512, 8000: 256}


def _chemin_dans_le_paquet() -> Path | None:
    """Localise le modèle livré avec ``silero-vad`` sans importer le paquet.

    ``find_spec`` n'exécute pas le module : c'est indispensable ici, puisque
    l'importer réclamerait torch — précisément ce qu'on évite.
    """
    try:
        spec = importlib.util.find_spec("silero_vad")
    except (ImportError, ValueError):  # pragma: no cover
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    chemin = Path(next(iter(spec.submodule_search_locations))) / "data" / "silero_vad.onnx"
    return chemin if chemin.exists() else None


class SileroVAD(VADEngine):
    name = "silero"

    def __init__(
        self,
        model_path: str | Path | None = None,
        sample_rate: int = 16000,
        models_dir: str | Path = "models/silero",
        num_threads: int = 1,
        **_ignored: Any,
    ) -> None:
        if sample_rate not in TAILLES_DE_TRAME:
            raise EngineUnavailable(
                f"Silero VAD accepte 8000 ou 16000 Hz, pas {sample_rate}."
            )
        self._sample_rate = sample_rate
        self._frame_size = TAILLES_DE_TRAME[sample_rate]
        self.models_dir = Path(models_dir).expanduser()
        self.explicit_path = Path(model_path).expanduser() if model_path else None
        self.num_threads = num_threads
        self._session: Any = None
        self._state: Any = None
        self._sr: Any = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return self._frame_size

    def model_path(self) -> Path | None:
        if self.explicit_path is not None:
            return self.explicit_path if self.explicit_path.exists() else None
        depuis_le_paquet = _chemin_dans_le_paquet()
        if depuis_le_paquet is not None:
            return depuis_le_paquet
        local = self.models_dir / "silero_vad.onnx"
        return local if local.exists() else None

    def available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return self.model_path() is not None

    def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dépend de l'install
            raise EngineUnavailable(
                "Le paquet « onnxruntime » est absent. "
                "Installez-le avec : pip install onnxruntime"
            ) from exc

        chemin = self.model_path()
        if chemin is None:
            raise EngineUnavailable(
                "Modèle Silero VAD introuvable. Le plus simple :\n"
                "  pip install --no-deps silero-vad\n"
                "(le fichier silero_vad.onnx est livré avec le paquet ; --no-deps "
                "évite d'installer torch, inutile ici). Sinon, déposez "
                f"silero_vad.onnx dans {self.models_dir}."
            )

        options = ort.SessionOptions()
        options.inter_op_num_threads = self.num_threads
        options.intra_op_num_threads = self.num_threads
        logger.info("Chargement du VAD Silero : %s", chemin)
        self._session = ort.InferenceSession(
            str(chemin), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(self._sample_rate, dtype=np.int64)
        self.reset()
        return self._session

    def reset(self) -> None:
        import numpy as np

        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def speech_probability(self, frame: bytes) -> float:
        import numpy as np

        session = self._get_session()
        echantillons = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        if echantillons.size != self._frame_size:
            raise EngineUnavailable(
                f"Silero attend exactement {self._frame_size} échantillons, "
                f"reçu {echantillons.size}. Passez par un Rechunker."
            )
        sortie, self._state = session.run(
            None,
            {
                "input": echantillons.reshape(1, -1),
                "state": self._state,
                "sr": self._sr,
            },
        )
        return float(sortie[0][0])

    def close(self) -> None:
        self._session = None

    def describe(self) -> str:
        return "silero(onnxruntime)"

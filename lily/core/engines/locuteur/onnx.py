"""Signature vocale par un vrai modèle de locuteur, en ONNX.

Le moteur spectral sépare des voix nettement différentes. Il ne sépare pas
deux voix proches — deux frères, deux collègues du même registre — parce que
des statistiques de spectre ne sont pas un modèle de locuteur. Quand cette
limite compte, c'est ici qu'on la lève.

Aucun modèle n'est téléchargé automatiquement, et aucune URL n'est codée en
dur : contrairement à Silero ou à Piper, il n'existe pas d'artefact unique et
stable qui fasse consensus pour le français. Vous fournissez donc le fichier,
et la configuration le désigne ::

    [voix]
    engine = "onnx"
    model_path = "~/.lily/models/locuteur.onnx"

Les familles qui conviennent, par ordre de robustesse décroissante et de
poids décroissant : **ECAPA-TDNN**, **ResNet de WeSpeaker**, **CAM++**. Toutes
prennent une forme d'onde 16 kHz ou un banc de filtres et rendent un vecteur
de 192 à 512 dimensions. Le moteur s'accommode des deux conventions d'entrée :
il regarde ce que le graphe déclare et lui donne ce qu'il demande.

``onnxruntime`` est déjà une dépendance optionnelle du projet — c'est lui qui
fait tourner Silero et openWakeWord. Ce moteur n'ajoute donc rien à installer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...audio import AudioBuffer
from ...errors import EngineUnavailable
from ...interfaces.locuteur import SpeakerEngine
from ...logging import get_logger

logger = get_logger("locuteur.onnx")

DUREE_MINIMALE_S = 0.5


class OnnxSpeaker(SpeakerEngine):
    """Un modèle de locuteur exporté en ONNX, fourni par vous."""

    name = "onnx"

    def __init__(
        self,
        model_path: str | Path = "",
        sample_rate: int = 16000,
        **_ignored: Any,
    ) -> None:
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.sample_rate = sample_rate
        self.model = self.model_path.stem if self.model_path else ""
        self._session: Any = None
        self._dimension = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return self.model_path is not None and self.model_path.is_file()

    def unavailable_reason(self) -> str:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return (
                "Le paquet « onnxruntime » est absent. Installez-le avec : "
                "pip install onnxruntime"
            )
        if self.model_path is None:
            return (
                "Aucun modèle de locuteur désigné. Renseignez voix.model_path "
                "avec un ECAPA-TDNN, un WeSpeaker ou un CAM++ exporté en ONNX."
            )
        if not self.model_path.is_file():
            return f"Modèle de locuteur introuvable : {self.model_path}"
        return ""

    def _charger(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - dépend de l'install
            raise EngineUnavailable(self.unavailable_reason()) from exc
        if self.model_path is None or not self.model_path.is_file():
            raise EngineUnavailable(self.unavailable_reason())
        logger.info("Chargement du modèle de locuteur %s", self.model_path.name)
        self._session = onnxruntime.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        sortie = self._session.get_outputs()[0]
        # La dernière dimension déclarée est celle du vecteur, quand elle est
        # fixe. Sinon on la découvrira à la première signature.
        derniere = sortie.shape[-1] if sortie.shape else None
        self._dimension = derniere if isinstance(derniere, int) else 0
        return self._session

    def encode(self, audio: AudioBuffer) -> list[float]:
        if not audio or audio.duration_s < DUREE_MINIMALE_S:
            return []
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy vient avec l'audio
            return []
        try:
            session = self._charger()
        except EngineUnavailable as exc:
            logger.warning("%s", exc)
            return []

        entree = session.get_inputs()[0]
        forme = entree.shape
        echantillons = audio.to_float32()
        # Deux conventions courantes : (lot, échantillons) pour un modèle qui
        # fait lui-même son banc de filtres, (lot, trames, bandes) pour un
        # modèle qui l'attend tout fait. On donne ce que le graphe demande.
        if len(forme) == 3:
            donnees = _banc_de_filtres(np, echantillons, self.sample_rate)[None, :, :]
        else:
            donnees = echantillons[None, :]

        try:
            sortie = session.run(None, {entree.name: donnees.astype(np.float32)})[0]
        except Exception:
            logger.exception("Le modèle de locuteur a refusé cette entrée.")
            return []

        vecteur = np.asarray(sortie).reshape(-1).astype(float)
        norme = float(np.linalg.norm(vecteur))
        if norme == 0.0 or not np.isfinite(norme):
            return []
        self._dimension = int(vecteur.size)
        return (vecteur / norme).tolist()

    def close(self) -> None:
        self._session = None


def _banc_de_filtres(np: Any, echantillons: Any, sample_rate: int, bandes: int = 80) -> Any:
    """Un banc de filtres mel, pour les modèles qui l'attendent tout fait.

    Volontairement minimal : les modèles de cette famille sont robustes aux
    petites différences de calcul, et reproduire exactement le banc de tel
    entraînement demanderait de le connaître.
    """
    from .spectral import _en_hertz, _en_mel

    taille = int(sample_rate * 0.025)
    pas = int(sample_rate * 0.010)
    if echantillons.size < taille:
        return np.zeros((1, bandes), dtype=np.float32)
    nombre = 1 + (echantillons.size - taille) // pas
    fenetres = np.lib.stride_tricks.as_strided(
        echantillons,
        shape=(nombre, taille),
        strides=(echantillons.strides[0] * pas, echantillons.strides[0]),
    ) * np.hanning(taille)

    spectre = np.abs(np.fft.rfft(fenetres, axis=1)) ** 2
    bornes = np.linspace(_en_mel(0.0), _en_mel(sample_rate / 2), bandes + 2)
    hertz = np.array([_en_hertz(m) for m in bornes])
    indices = np.floor((taille + 1) * hertz / sample_rate).astype(int)
    banc = np.zeros((bandes, taille // 2 + 1))
    for filtre in range(bandes):
        gauche, sommet, droite = indices[filtre : filtre + 3]
        for k in range(gauche, min(sommet, banc.shape[1])):
            if sommet > gauche:
                banc[filtre, k] = (k - gauche) / (sommet - gauche)
        for k in range(sommet, min(droite, banc.shape[1])):
            if droite > sommet:
                banc[filtre, k] = (droite - k) / (droite - sommet)
    return np.log(spectre @ banc.T + 1e-10)

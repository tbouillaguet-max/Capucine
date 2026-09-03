"""Un seul point d'entrée et de sortie audio.

Deux décisions portent tout le module :

* **Le transport parle en PCM entier 16 bits mono**, pas en tableaux numpy.
  C'est le format natif de ``sounddevice`` comme de ``piper`` ; la conversion
  en flottants n'a lieu qu'à la frontière de la transcription, qui dépend de
  toute façon de numpy. Ce module reste donc sans dépendance : il s'importe
  sur une machine nue.
* **``sounddevice`` est importé paresseusement.** Toutes les implémentations
  en mémoire (``MemoryAudioInput``, ``MemoryAudioOutput``, ``WavFileInput``,
  ``WavFileOutput``) fonctionnent sans micro, sans haut-parleur et sans
  PortAudio installé — c'est ce qui rend la chaîne vocale testable.

La lecture consulte un ``threading.Event`` entre deux tranches : c'est le
mécanisme sur lequel le barge-in de l'étape 3 viendra se brancher.
"""

from __future__ import annotations

import queue
import threading
import wave
from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typage seul
    import numpy as np

logger = get_logger("audio")

SAMPLE_WIDTH = 2  # entiers signés 16 bits
CHANNELS = 1


@dataclass(frozen=True)
class AudioBuffer:
    """Un extrait audio capté, en PCM 16 bits mono petit-boutiste."""

    pcm: bytes
    sample_rate: int

    @property
    def n_samples(self) -> int:
        return len(self.pcm) // SAMPLE_WIDTH

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate if self.sample_rate else 0.0

    def __bool__(self) -> bool:
        return bool(self.pcm)

    def to_float32(self) -> np.ndarray:
        """Convertit en flottants dans [-1, 1], format attendu par Whisper.

        numpy n'est importé qu'ici : le transport audio ne le réclame jamais.
        """
        import numpy as np

        return np.frombuffer(self.pcm, dtype=np.int16).astype(np.float32) / 32768.0

    def rms(self) -> float:
        """Niveau sonore moyen, entre 0 et 1. Sert au repli d'énergie du VAD
        et au journal de diagnostic. Sans ``audioop``, retiré de Python 3.13."""
        if not self.pcm:
            return 0.0
        echantillons = array("h")
        echantillons.frombytes(self.pcm[: len(self.pcm) - len(self.pcm) % SAMPLE_WIDTH])
        somme = sum(float(v) * v for v in echantillons)
        return (somme / len(echantillons)) ** 0.5 / 32768.0


@dataclass
class AudioChunk:
    """Un morceau de parole synthétisée, prêt à être joué.

    Piper produit un morceau **par phrase** : c'est l'unité naturelle du
    streaming, et celle à laquelle on peut s'interrompre proprement.
    """

    pcm: bytes
    sample_rate: int
    text: str = ""

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / SAMPLE_WIDTH / self.sample_rate if self.sample_rate else 0.0


# --- entrée ----------------------------------------------------------------

class AudioInput(ABC):
    """Source de trames audio."""

    name: str = "entree"

    def available(self) -> bool:
        """Le périphérique est-il réellement utilisable ? Ne lève jamais."""
        return True

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def frame_size(self) -> int:
        """Nombre d'échantillons par trame."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def frames(self) -> Iterator[bytes]:
        """Trames PCM successives. Bloque jusqu'à ``stop()``."""

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> AudioInput:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class SoundDeviceInput(AudioInput):
    """Micro réel. ``sounddevice`` n'est importé qu'au démarrage du flux."""

    name = "sounddevice"

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        device: str | int | None = None,
        queue_frames: int = 200,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_size = int(sample_rate * frame_ms / 1000)
        self.device = device or None
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=queue_frames)
        self._stream: Any = None
        self._perdues = 0

    def available(self) -> bool:
        return _peripherique_disponible(self.device, "input")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return self._frame_size

    def _callback(self, indata: Any, _frames: int, _time: Any, status: Any) -> None:
        if status:
            logger.debug("état du flux d'entrée : %s", status)
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            # Mieux vaut perdre la trame la plus ancienne que bloquer le
            # thread temps réel de PortAudio.
            self._perdues += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(bytes(indata))
            except (queue.Empty, queue.Full):  # pragma: no cover
                pass

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise AudioUnavailable(
                "Entrée audio indisponible : « sounddevice » (et PortAudio) sont requis. "
                f"Installez-les avec : pip install sounddevice  ({exc})"
            ) from exc
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._frame_size,
            device=self.device,
            channels=CHANNELS,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        logger.debug("Micro ouvert : %d Hz, trames de %d échantillons",
                     self._sample_rate, self._frame_size)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._perdues:
            logger.warning("%d trames d'entrée perdues (machine surchargée ?)", self._perdues)
            self._perdues = 0
        self._queue.put_nowait(None) if not self._queue.full() else None

    def frames(self) -> Iterator[bytes]:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            yield frame


class MemoryAudioInput(AudioInput):
    """Source de trames préenregistrées. C'est la sortie « sans micro »."""

    name = "memoire"

    def __init__(
        self,
        frames: list[bytes],
        sample_rate: int = 16000,
        frame_size: int = 480,
        repeat: bool = False,
        frame_delay_s: float = 0.0,
    ) -> None:
        self._frames = list(frames)
        self._sample_rate = sample_rate
        self._frame_size = frame_size
        # `repeat` fait tourner la boucle jusqu'à `stop()` : c'est ce qu'il
        # faut pour éprouver une écoute permanente, où le micro ne se tarit
        # jamais. `frame_delay_s` rend la main au reste du programme.
        self.repeat = repeat
        self.frame_delay_s = frame_delay_s
        self._arrete = threading.Event()

    @classmethod
    def from_buffer(cls, buffer: AudioBuffer, frame_ms: int = 30) -> MemoryAudioInput:
        taille = int(buffer.sample_rate * frame_ms / 1000) * SAMPLE_WIDTH
        trames = [buffer.pcm[i : i + taille] for i in range(0, len(buffer.pcm), taille)]
        return cls(trames, buffer.sample_rate, taille // SAMPLE_WIDTH)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return self._frame_size

    def start(self) -> None:
        self._arrete.clear()

    def stop(self) -> None:
        self._arrete.set()

    def frames(self) -> Iterator[bytes]:
        while True:
            for frame in self._frames:
                if self._arrete.is_set():
                    return
                yield frame
                if self.frame_delay_s:
                    self._arrete.wait(self.frame_delay_s)
            if not self.repeat:
                return


class WavFileInput(MemoryAudioInput):
    """Rejoue un fichier WAV comme s'il venait du micro.

    C'est ce qui permet de tester la transcription de bout en bout, en
    intégration continue, sans le moindre périphérique.
    """

    name = "wav"

    def __init__(self, path: str | Path, frame_ms: int = 30) -> None:  # noqa: D107
        with wave.open(str(path), "rb") as handle:
            if handle.getsampwidth() != SAMPLE_WIDTH or handle.getnchannels() != CHANNELS:
                raise AudioUnavailable(
                    f"{path} doit être en PCM 16 bits mono "
                    f"(reçu {handle.getsampwidth() * 8} bits, {handle.getnchannels()} canaux)."
                )
            buffer = AudioBuffer(handle.readframes(handle.getnframes()), handle.getframerate())
        super().__init__(*_decouper(buffer, frame_ms))


def _decouper(buffer: AudioBuffer, frame_ms: int) -> tuple[list[bytes], int, int]:
    taille = int(buffer.sample_rate * frame_ms / 1000) * SAMPLE_WIDTH
    trames = [buffer.pcm[i : i + taille] for i in range(0, len(buffer.pcm), taille)]
    return trames, buffer.sample_rate, taille // SAMPLE_WIDTH


# --- sortie ----------------------------------------------------------------

class AudioOutput(ABC):
    """Destination de la parole synthétisée."""

    name: str = "sortie"

    def available(self) -> bool:
        """Le périphérique est-il réellement utilisable ? Ne lève jamais."""
        return True

    @abstractmethod
    def play(self, chunk: AudioChunk, cancel: threading.Event | None = None) -> bool:
        """Joue un morceau. Retourne ``False`` s'il a été interrompu."""

    def stop(self) -> None:
        """Coupe la lecture en cours, immédiatement."""

    def close(self) -> None:
        self.stop()


class SoundDeviceOutput(AudioOutput):
    """Haut-parleur réel.

    Le flux est (ré)ouvert à la fréquence du morceau : le micro tourne à
    16 kHz, les voix Piper à 22,05 kHz, et rien n'oblige à rééchantillonner.
    L'écriture se fait par tranches de quelques dizaines de millisecondes, en
    consultant ``cancel`` entre chacune, pour que l'interruption soit franche.
    """

    name = "sounddevice"

    def __init__(self, device: str | int | None = None, slice_ms: int = 40) -> None:
        self.device = device or None
        self.slice_ms = slice_ms
        self._stream: Any = None
        self._sample_rate: int | None = None
        self._stop = threading.Event()

    def available(self) -> bool:
        return _peripherique_disponible(self.device, "output")

    def _ensure_stream(self, sample_rate: int) -> Any:
        if self._stream is not None and self._sample_rate == sample_rate:
            return self._stream
        self._close_stream()
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise AudioUnavailable(
                "Sortie audio indisponible : « sounddevice » (et PortAudio) sont requis. "
                f"Installez-les avec : pip install sounddevice  ({exc})"
            ) from exc
        self._stream = sd.RawOutputStream(
            samplerate=sample_rate, device=self.device, channels=CHANNELS, dtype="int16"
        )
        self._stream.start()
        self._sample_rate = sample_rate
        return self._stream

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover - fermeture au mieux
                logger.debug("Fermeture du flux de sortie en erreur.", exc_info=True)
        self._stream = None
        self._sample_rate = None

    def play(self, chunk: AudioChunk, cancel: threading.Event | None = None) -> bool:
        if not chunk.pcm:
            return True
        self._stop.clear()
        stream = self._ensure_stream(chunk.sample_rate)
        taille = max(1, int(chunk.sample_rate * self.slice_ms / 1000)) * SAMPLE_WIDTH
        for debut in range(0, len(chunk.pcm), taille):
            if self._stop.is_set() or (cancel is not None and cancel.is_set()):
                try:
                    stream.abort()
                except Exception:  # pragma: no cover
                    logger.debug("Abandon du flux en erreur.", exc_info=True)
                self._close_stream()
                return False
            stream.write(chunk.pcm[debut : debut + taille])
        return True

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()
        self._close_stream()


class MemoryAudioOutput(AudioOutput):
    """Haut-parleur factice qui garde tout ce qu'on lui a donné."""

    name = "memoire"

    def __init__(self) -> None:
        self.chunks: list[AudioChunk] = []
        self.interrompus: list[AudioChunk] = []

    def play(self, chunk: AudioChunk, cancel: threading.Event | None = None) -> bool:
        if cancel is not None and cancel.is_set():
            self.interrompus.append(chunk)
            return False
        self.chunks.append(chunk)
        return True

    @property
    def pcm(self) -> bytes:
        return b"".join(chunk.pcm for chunk in self.chunks)

    @property
    def texte(self) -> str:
        return " ".join(chunk.text for chunk in self.chunks if chunk.text).strip()


@dataclass
class WavFileOutput(AudioOutput):
    """Écrit la parole dans un fichier WAV, pour écouter après coup ce que
    Lily a dit sans avoir de haut-parleur sous la main."""

    path: Path
    name: str = "wav"
    _chunks: list[AudioChunk] = field(default_factory=list)

    def play(self, chunk: AudioChunk, cancel: threading.Event | None = None) -> bool:
        if cancel is not None and cancel.is_set():
            return False
        self._chunks.append(chunk)
        self._ecrire()
        return True

    def _ecrire(self) -> None:
        if not self._chunks:
            return
        with wave.open(str(self.path), "wb") as handle:
            handle.setnchannels(CHANNELS)
            handle.setsampwidth(SAMPLE_WIDTH)
            handle.setframerate(self._chunks[0].sample_rate)
            for chunk in self._chunks:
                handle.writeframes(chunk.pcm)


class NullAudioOutput(AudioOutput):
    """Ne joue rien. Utile pour mesurer la latence sans le temps de lecture."""

    name = "silence"

    def play(self, chunk: AudioChunk, cancel: threading.Event | None = None) -> bool:
        return not (cancel is not None and cancel.is_set())


# --- regroupement de trames ------------------------------------------------

class Rechunker:
    """Regroupe des trames de taille quelconque en trames de taille fixe.

    Chaque étage réclame sa propre découpe : Silero exige exactement 512
    échantillons, openWakeWord travaille par 1280, et le micro délivre ce que
    demande la configuration. Plutôt que d'imposer une taille commune — qui
    serait mauvaise pour tout le monde — chaque consommateur a son rechunker.
    """

    __slots__ = ("frame_bytes", "_tampon")

    def __init__(self, frame_size: int) -> None:
        if frame_size <= 0:
            raise ValueError("La taille de trame doit être strictement positive.")
        self.frame_bytes = frame_size * SAMPLE_WIDTH
        self._tampon = bytearray()

    def push(self, pcm: bytes) -> Iterator[bytes]:
        self._tampon.extend(pcm)
        while len(self._tampon) >= self.frame_bytes:
            yield bytes(self._tampon[: self.frame_bytes])
            del self._tampon[: self.frame_bytes]

    def reset(self) -> None:
        self._tampon.clear()

    @property
    def en_attente(self) -> int:
        return len(self._tampon)


# --- utilitaires -----------------------------------------------------------

class AudioUnavailable(RuntimeError):
    """Périphérique ou bibliothèque audio absente. Message toujours explicite."""


def _peripherique_disponible(device: str | int | None, sens: str) -> bool:
    """PortAudio est-il là, et existe-t-il un périphérique de ce sens ?

    Vérifié au démarrage plutôt qu'à la première phrase : mieux vaut annoncer
    tout de suite « pas de haut-parleur, j'afficherai » qu'échouer à chaque
    phrase prononcée.
    """
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        logger.debug("sounddevice indisponible : %s", exc)
        return False
    try:
        sd.query_devices(device, sens)
    except Exception as exc:  # noqa: BLE001 - PortAudio lève des types variés
        logger.debug("Aucun périphérique %s utilisable : %s", sens, exc)
        return False
    return True


def record(
    source: AudioInput,
    *,
    stop: threading.Event | None = None,
    max_seconds: float = 20.0,
) -> AudioBuffer:
    """Capte jusqu'à ``stop`` ou ``max_seconds``, selon ce qui vient en premier.

    C'est la primitive du « appuie pour parler » de l'étape 2 ; l'étape 3 y
    branchera le VAD en passant un ``stop`` armé par la détection de silence.
    """
    limite = int(max_seconds * source.sample_rate) * SAMPLE_WIDTH
    morceaux: list[bytes] = []
    total = 0
    for frame in source.frames():
        morceaux.append(frame)
        total += len(frame)
        if total >= limite:
            logger.debug("Durée maximale de capture atteinte (%.1f s).", max_seconds)
            break
        if stop is not None and stop.is_set():
            break
    return AudioBuffer(b"".join(morceaux), source.sample_rate)


def list_devices() -> str:
    """Inventaire des périphériques, pour renseigner audio.input_device."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        return (
            "Périphériques indisponibles : « sounddevice » (et PortAudio) ne sont pas "
            f"installés.\nInstallez-les avec : pip install sounddevice\n({exc})"
        )
    return str(sd.query_devices())

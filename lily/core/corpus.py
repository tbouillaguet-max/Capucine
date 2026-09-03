"""Le corpus d'éveil : apprendre votre voix, pas une voix moyenne.

Le modèle « lily » livré est entraîné sur des voix de synthèse
françaises. Il marche, mais il ne connaît ni votre timbre, ni votre débit, ni
l'acoustique de votre pièce — et c'est exactement ce qui fait la différence
entre un mot d'éveil qui répond au premier coup et un qu'il faut répéter.

D'où ce mécanisme, qui est le seul des six à préparer un vrai réentraînement.
Il ne réentraîne rien tout seul : il **collecte les exemples étiquetés** que
``tools/entrainer_lily.py`` réclame, et qu'on ne peut obtenir qu'en usage
réel.

L'étiquetage se fait sans rien vous demander, à partir de ce qui se passe
juste après le déclenchement :

* un énoncé transcrit derrière un éveil → **vrai positif**, elle a eu raison ;
* un éveil suivi de rien du tout → **faux positif** probable, la télévision
  ou une conversation ont suffi à la réveiller.

Trois précautions, parce qu'écrire du son sur un disque n'est pas anodin :

* **Désactivé par défaut.** ``[corpus] actif = true`` est une décision qui
  vous appartient, pas un défaut que vous découvririez après coup.
* **Seule la fenêtre du mot d'éveil est gardée** — environ deux secondes qui
  se terminent juste après « Lily ». Ce que vous dites ensuite n'est
  jamais enregistré.
* **Le nombre de fichiers est borné**, les plus anciens partent. Un corpus
  d'éveil ne doit pas se transformer en archive de votre salon.

Rien ne sort de la machine, ici comme ailleurs.
"""

from __future__ import annotations

import threading
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .logging import get_logger

logger = get_logger("corpus")

EVEILS = "eveils"
FAUX_POSITIFS = "faux_positifs"
EN_ATTENTE = "en_attente"
MINIMUM_SECONDES = 0.2


@dataclass
class EtatDuCorpus:
    eveils: int
    faux_positifs: int
    en_attente: int
    dossier: Path

    @property
    def total(self) -> int:
        return self.eveils + self.faux_positifs

    @property
    def taux_de_faux_positifs(self) -> float:
        return self.faux_positifs / self.total if self.total else 0.0


class CorpusEveil:
    """Garde la fenêtre sonore autour de chaque déclenchement, et l'étiquette."""

    def __init__(
        self,
        dossier: str | Path,
        *,
        actif: bool = False,
        sample_rate: int = 16000,
        secondes_avant: float = 1.6,
        secondes_apres: float = 0.4,
        maximum_par_classe: int = 300,
    ) -> None:
        self.dossier = Path(dossier).expanduser()
        self.actif = actif
        self.sample_rate = sample_rate
        self.secondes_avant = secondes_avant
        self.secondes_apres = secondes_apres
        self.maximum_par_classe = maximum_par_classe

        self._verrou = threading.Lock()
        # Anneau d'octets PCM : on le dimensionne en octets, pas en trames,
        # pour ne rien supposer de la taille des trames du micro.
        self._octets_gardes = int(secondes_avant * sample_rate * 2)
        self._anneau: deque[bytes] = deque()
        self._taille = 0
        # Après un déclenchement, on continue d'accumuler un court instant :
        # la fin du mot arrive souvent après la détection.
        self._queue_restante = 0
        self._en_cours: bytearray | None = None
        self._dernier: Path | None = None
        self._score = 0.0
        # Deux captures dans la même milliseconde ne doivent pas se marcher
        # dessus : l'horodatage seul ne suffit pas à nommer un fichier.
        self._numero = 0

    # -- alimentation, depuis le fil d'écoute -------------------------------
    def alimenter(self, frame: bytes) -> None:
        """Reçoit une trame du mode « éveil ». Appelé très souvent : pas de
        disque, pas d'allocation inutile, jamais d'exception."""
        if not self.actif or not frame:
            return
        with self._verrou:
            if self._en_cours is not None:
                self._prolonger(frame)
                return
            self._anneau.append(frame)
            self._taille += len(frame)
            while self._taille > self._octets_gardes and len(self._anneau) > 1:
                self._taille -= len(self._anneau.popleft())

    def completer(self, frame: bytes) -> None:
        """Prolonge une fenêtre déjà déclenchée, et rien d'autre.

        C'est ce que le fil d'écoute appelle une fois passé en mode énoncé :
        la fin du mot d'éveil arrive après la détection, mais **ce que vous
        dites ensuite ne doit jamais entrer dans l'anneau**. D'où deux
        méthodes plutôt qu'une : celle-ci ne peut pas se mettre à enregistrer.
        """
        if not self.actif or not frame:
            return
        with self._verrou:
            if self._en_cours is not None:
                self._prolonger(frame)

    def _prolonger(self, frame: bytes) -> None:
        """Ajoute la trame à la queue en cours. Verrou déjà tenu."""
        manquant = self._queue_restante
        self._en_cours.extend(frame[:manquant] if len(frame) > manquant else frame)
        self._queue_restante -= len(frame)
        if self._queue_restante <= 0:
            self._ecrire_en_attente()

    def declencher(self, score: float = 0.0) -> None:
        """Un éveil vient d'être détecté : on fige la fenêtre qui précède.

        L'écriture n'a pas lieu ici mais quelques trames plus tard, quand la
        queue est complète — sans quoi on couperait la dernière syllabe.
        """
        if not self.actif:
            return
        with self._verrou:
            if self._en_cours is not None:
                return
            self._en_cours = bytearray(b"".join(self._anneau))
            self._queue_restante = max(1, int(self.secondes_apres * self.sample_rate * 2))
            self._score = score
            self._anneau.clear()
            self._taille = 0

    def cloturer(self) -> Path | None:
        """Écrit tout de suite ce qui est en cours, sans attendre la queue.

        Utile quand l'écoute change de mode juste après le déclenchement : la
        fenêtre serait sinon perdue.
        """
        with self._verrou:
            if self._en_cours is None:
                return None
            return self._ecrire_en_attente()

    def _ecrire_en_attente(self) -> Path | None:
        """Écrit la fenêtre courante dans « en_attente ». Verrou déjà tenu."""
        donnees = bytes(self._en_cours or b"")
        self._en_cours = None
        self._queue_restante = 0
        # Moins de deux dixièmes de seconde : ce n'est pas un mot d'éveil, et
        # un extrait trop court dégraderait l'entraînement au lieu de l'aider.
        # (Deux octets par échantillon : c'est du PCM 16 bits mono.)
        if len(donnees) < MINIMUM_SECONDES * self.sample_rate * 2:
            return None
        chemin = self._chemin(EN_ATTENTE, self._score)
        try:
            _ecrire_wav(chemin, donnees, self.sample_rate)
        except OSError:
            logger.exception("Écriture du corpus d'éveil impossible.")
            return None
        self._dernier = chemin
        return chemin

    # -- étiquetage ---------------------------------------------------------
    def confirmer(self) -> Path | None:
        """Le dernier éveil était bon : la fenêtre devient un positif."""
        return self._classer(EVEILS)

    def dementir(self) -> Path | None:
        """Le dernier éveil était faux : la fenêtre devient un négatif.

        Ce sont les exemples les plus précieux du lot. Un modèle de mot
        d'éveil ne souffre presque jamais d'un manque de positifs ; il souffre
        de faux déclenchements, et ceux-là ne s'inventent pas en studio.
        """
        return self._classer(FAUX_POSITIFS)

    def _classer(self, classe: str) -> Path | None:
        with self._verrou:
            if self._en_cours is not None:
                self._ecrire_en_attente()
            source = self._dernier
            self._dernier = None
        if source is None or not source.exists():
            return None
        cible = self.dossier / classe / source.name
        cible.parent.mkdir(parents=True, exist_ok=True)
        try:
            source.replace(cible)
        except OSError:
            logger.exception("Classement d'un extrait d'éveil impossible.")
            return None
        logger.info("Corpus d'éveil : %s → %s", source.name, classe)
        self._elaguer(classe)
        return cible

    def _elaguer(self, classe: str) -> None:
        """Garde les plus récents : un corpus borné, pas une archive."""
        dossier = self.dossier / classe
        # Par date de fichier, pas par nom : le compteur du nom repart à zéro
        # à chaque démarrage, la date du fichier non.
        fichiers = sorted(dossier.glob("*.wav"), key=lambda f: f.stat().st_mtime)
        for vieux in fichiers[: max(0, len(fichiers) - self.maximum_par_classe)]:
            try:
                vieux.unlink()
            except OSError:  # pragma: no cover
                logger.debug("Suppression de %s impossible.", vieux)

    def oublier_les_en_attente(self) -> int:
        """Jette les fenêtres qu'aucun tour n'a étiquetées."""
        dossier = self.dossier / EN_ATTENTE
        nombre = 0
        for fichier in dossier.glob("*.wav"):
            try:
                fichier.unlink()
                nombre += 1
            except OSError:  # pragma: no cover
                logger.debug("Suppression de %s impossible.", fichier)
        with self._verrou:
            self._dernier = None
        return nombre

    def tout_oublier(self) -> int:
        nombre = 0
        for classe in (EVEILS, FAUX_POSITIFS, EN_ATTENTE):
            for fichier in (self.dossier / classe).glob("*.wav"):
                try:
                    fichier.unlink()
                    nombre += 1
                except OSError:  # pragma: no cover
                    logger.debug("Suppression de %s impossible.", fichier)
        with self._verrou:
            self._dernier = None
        return nombre

    # -- état ---------------------------------------------------------------
    def etat(self) -> EtatDuCorpus:
        return EtatDuCorpus(
            eveils=_compter(self.dossier / EVEILS),
            faux_positifs=_compter(self.dossier / FAUX_POSITIFS),
            en_attente=_compter(self.dossier / EN_ATTENTE),
            dossier=self.dossier,
        )

    def _chemin(self, classe: str, score: float) -> Path:
        dossier = self.dossier / classe
        dossier.mkdir(parents=True, exist_ok=True)
        horodatage = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._numero += 1
        return dossier / (
            f"{horodatage}-{self._numero:04d}_s{int(round(score * 100)):03d}.wav"
        )


def _compter(dossier: Path) -> int:
    try:
        return sum(1 for _ in dossier.glob("*.wav"))
    except OSError:  # pragma: no cover
        return 0


def _ecrire_wav(chemin: Path, pcm: bytes, sample_rate: int) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(chemin), "wb") as fichier:
        fichier.setnchannels(1)
        fichier.setsampwidth(2)
        fichier.setframerate(sample_rate)
        fichier.writeframes(pcm)


def depuis_config(config) -> CorpusEveil | None:
    """Construit le corpus décrit par ``[corpus]``.

    Renvoie toujours un objet quand la section existe : c'est le drapeau
    ``actif`` qui décide d'écrire ou non, pour que les compétences
    d'introspection puissent expliquer comment l'allumer.
    """
    section = config.section("corpus")
    dossier = (
        config.resolve_path("corpus.dossier")
        or Path.home() / ".lily" / "corpus"
    )
    return CorpusEveil(
        dossier,
        actif=bool(section.get("actif", False)),
        sample_rate=int(config.get("audio.sample_rate", 16000)),
        secondes_avant=float(section.get("secondes_avant", 1.6)),
        secondes_apres=float(section.get("secondes_apres", 0.4)),
        maximum_par_classe=int(section.get("maximum_par_classe", 300)),
    )

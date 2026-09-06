"""Ce que Lily sait de VOTRE voix.

C'est le septième mécanisme d'apprentissage, et le seul qui apprenne une
personne plutôt qu'une habitude. Le principe tient en trois temps :

1. **Enrôler.** Vous dites « apprends ma voix » quelques fois. Chaque énoncé
   devient une signature ; leur moyenne est le **centroïde**, la description
   de votre voix telle que ce micro l'entend.
2. **Reconnaître.** À chaque éveil, l'énoncé qui suit est signé et comparé au
   centroïde. En dessous du seuil, le tour est abandonné : la télévision a dit
   « Lily », pas vous.
3. **Suivre.** Un énoncé reconnu **largement** au-dessus du seuil rejoint les
   échantillons. Votre voix du matin n'est pas celle du soir, une pièce change
   quand on y met un tapis : le centroïde suit, sans qu'on redemande rien.

Quatre précautions, parce qu'un filtre qui décide « ce n'est pas vous » peut
vous enfermer dehors :

* **Éteint par défaut.** ``[voix] actif = true`` est une décision.
* **Sans enrôlement, il ne bloque rien.** Tant que le centroïde n'existe pas,
  ``reconnait()`` dit oui à tout le monde plutôt que non à tout le monde.
* **Le clavier n'est jamais filtré.** Une phrase tapée passe toujours ; c'est
  la porte de secours si le seuil est mal réglé.
* **La dérive est bornée.** L'apprentissage continu n'accepte un échantillon
  qu'au-delà du seuil PLUS une marge. Sans cela, chaque voix qui passe de
  justesse tirerait le centroïde vers elle, et de proche en proche le filtre
  finirait par accepter tout le monde — un mode d'échec silencieux, et le
  seul vraiment dangereux ici.

Ce n'est pas de l'authentification. Une signature vocale se trompe, et un
enregistrement de votre voix la trompe. Elle écarte le bruit d'une pièce ;
elle ne garde pas un secret.
"""

from __future__ import annotations

import array
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .audio import AudioBuffer
from .interfaces.locuteur import SpeakerEngine
from .logging import get_logger
from .sqlite import regler_la_base

logger = get_logger("locuteur")

SCHEMA = """
CREATE TABLE IF NOT EXISTS empreintes_vocales (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    personne    TEXT NOT NULL DEFAULT 'moi',
    moteur      TEXT NOT NULL,          -- deux moteurs = deux espaces incomparables
    dimension   INTEGER NOT NULL,
    vecteur     BLOB NOT NULL,
    origine     TEXT NOT NULL,          -- enrolement | usage
    horodatage  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_empreintes_personne
    ON empreintes_vocales(personne, moteur);
"""


@dataclass
class Verdict:
    """Ce que rend une comparaison. Toujours lisible, jamais un simple booléen."""

    reconnu: bool
    score: float
    raison: str

    def __bool__(self) -> bool:
        return self.reconnu


@dataclass
class EtatDeLaVoix:
    echantillons: int
    moteur: str
    seuil: float
    actif: bool
    enrole: bool

    def decrire(self) -> str:
        if not self.actif:
            return "reconnaissance de la voix éteinte ([voix] actif = false)"
        if not self.enrole:
            return (
                f"aucune voix apprise ({self.echantillons} échantillon(s) sur "
                f"le minimum requis) — dites « apprends ma voix »"
            )
        return (
            f"{self.echantillons} échantillon(s), moteur {self.moteur}, "
            f"seuil {self.seuil:.2f}"
        )


class Locuteur:
    """Le magasin des signatures vocales, et la décision qui va avec."""

    def __init__(
        self,
        chemin: str | Path,
        moteur: SpeakerEngine | None = None,
        *,
        actif: bool = False,
        seuil: float = 0.62,
        echantillons_min: int = 3,
        echantillons_max: int = 40,
        apprentissage_continu: bool = True,
        marge_d_apprentissage: float = 0.10,
        personne: str = "moi",
    ) -> None:
        self.chemin = Path(chemin).expanduser()
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self.moteur = moteur
        self.actif = actif
        self.seuil = seuil
        self.echantillons_min = echantillons_min
        self.echantillons_max = echantillons_max
        self.apprentissage_continu = apprentissage_continu
        self.marge_d_apprentissage = marge_d_apprentissage
        self.personne = personne

        self._verrou = threading.RLock()
        self._db = sqlite3.connect(str(self.chemin), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        regler_la_base(self._db)
        with self._verrou:
            self._db.executescript(SCHEMA)
            self._db.commit()
        self._centroide: list[float] | None = None
        self._compte: int | None = None
        # Le dernier énoncé entendu, que la compétence d'enrôlement reprend.
        # Sans lui, « apprends ma voix » n'aurait rien à signer : un plugin ne
        # voit pas l'audio, il ne voit que du texte.
        self.derniere_capture: AudioBuffer | None = None

    def fermer(self) -> None:
        with self._verrou:
            self._db.close()
        if self.moteur is not None:
            self.moteur.close()

    # -- état ---------------------------------------------------------------
    @property
    def disponible(self) -> bool:
        return self.moteur is not None

    @property
    def enrole(self) -> bool:
        """Y a-t-il assez d'échantillons pour que le filtre ait un sens ?"""
        return self._nombre() >= self.echantillons_min

    @property
    def opere(self) -> bool:
        """Le filtre est-il en état de refuser quelqu'un ?

        Trois conditions, et l'ordre compte : allumé, un moteur, et enrôlé.
        Il manque l'une des trois et tout passe — refuser sans savoir
        reconnaître serait la pire des façons d'échouer.
        """
        return self.actif and self.disponible and self.enrole

    def etat(self) -> EtatDeLaVoix:
        return EtatDeLaVoix(
            echantillons=self._nombre(),
            moteur=self.moteur.describe() if self.moteur else "aucun",
            seuil=self.seuil,
            actif=self.actif,
            enrole=self.enrole,
        )

    # -- enrôlement ---------------------------------------------------------
    def enroler(self, audio: AudioBuffer | None = None, *, origine: str = "enrolement") -> Verdict:
        """Ajoute un extrait aux échantillons de référence.

        Sans argument, reprend le dernier énoncé entendu — c'est ce que fait
        « apprends ma voix » : vous venez de parler, il n'y a pas de raison de
        vous faire parler une seconde fois.
        """
        audio = audio if audio is not None else self.derniere_capture
        if self.moteur is None:
            return Verdict(False, 0.0, "aucun moteur de signature vocale n'est disponible")
        if audio is None or not audio:
            # Le cas le plus fréquent, et il n'a rien d'une panne : on a tapé
            # la phrase au lieu de la dire. Une signature vocale a besoin de
            # voix.
            return Verdict(
                False, 0.0,
                "je n'ai rien entendu à signer : dites-le au micro plutôt qu'au clavier",
            )

        signature = self._signer(audio)
        if not signature:
            return Verdict(
                False, 0.0,
                "cet extrait ne contient pas assez de parole pour en tirer une signature",
            )

        with self._verrou:
            self._db.execute(
                """INSERT INTO empreintes_vocales
                   (personne, moteur, dimension, vecteur, origine, horodatage)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.personne, self._nom_du_moteur(), len(signature),
                 _en_blob(signature), origine, _maintenant()),
            )
            self._elaguer()
            self._db.commit()
            self._centroide = None
            self._compte = None

        nombre = self._nombre()
        if nombre < self.echantillons_min:
            manque = self.echantillons_min - nombre
            return Verdict(
                True, 0.0,
                f"c'est noté — encore {manque} fois et je saurai vous reconnaître",
            )
        return Verdict(True, 0.0, f"je reconnais votre voix ({nombre} échantillons)")

    def oublier(self) -> int:
        """Efface tout ce qui a été appris de cette voix."""
        with self._verrou:
            nombre = self._db.execute(
                "DELETE FROM empreintes_vocales WHERE personne = ?", (self.personne,)
            ).rowcount
            self._db.commit()
            self._centroide = None
            self._compte = None
        logger.info("Voix oubliée : %d échantillon(s) effacé(s).", nombre)
        return nombre

    # -- reconnaissance -----------------------------------------------------
    def reconnait(self, audio: AudioBuffer | None) -> Verdict:
        """Est-ce la personne enrôlée qui vient de parler ?

        Ne lève jamais, et **dit oui en cas de doute** : tant que le filtre
        n'est pas en état d'opérer, il laisse passer. Un assistant sourd à son
        propriétaire est plus grave qu'un assistant qui répond à un invité.
        """
        if not self.opere:
            return Verdict(True, 0.0, "filtre inactif")
        if audio is None or not audio:
            return Verdict(True, 0.0, "rien à comparer")

        try:
            signature = self._signer(audio)
            centroide = self._centroide_courant()
        except Exception:  # pragma: no cover - une signature ne casse pas un tour
            logger.exception("Comparaison de voix impossible ; on laisse passer.")
            return Verdict(True, 0.0, "comparaison impossible")

        if not signature or not centroide:
            return Verdict(True, 0.0, "signature indisponible")

        score = _produit_scalaire(signature, centroide)
        if score < self.seuil:
            logger.info("Voix non reconnue (%.3f < %.2f) : tour abandonné.", score, self.seuil)
            return Verdict(False, score, "cette voix n'est pas celle que j'ai apprise")

        if self.apprentissage_continu and score >= self.seuil + self.marge_d_apprentissage:
            # Franchement reconnu : on en profite pour suivre la voix. La marge
            # est ce qui empêche le centroïde de dériver vers qui passe de
            # justesse — et de finir par accepter tout le monde.
            self._retenir(signature)
        return Verdict(True, score, "voix reconnue")

    # -- interne ------------------------------------------------------------
    def _signer(self, audio: AudioBuffer) -> list[float]:
        assert self.moteur is not None
        return self.moteur.encode(audio)

    def _nom_du_moteur(self) -> str:
        return self.moteur.describe() if self.moteur else "aucun"

    def _nombre(self) -> int:
        if self._compte is not None:
            return self._compte
        if self.moteur is None:
            return 0
        with self._verrou:
            self._compte = self._db.execute(
                "SELECT COUNT(*) FROM empreintes_vocales WHERE personne = ? AND moteur = ?",
                (self.personne, self._nom_du_moteur()),
            ).fetchone()[0]
        return self._compte

    def _centroide_courant(self) -> list[float]:
        """La moyenne des signatures, renormalisée. Servie depuis un cache."""
        if self._centroide is not None:
            return self._centroide
        with self._verrou:
            lignes = self._db.execute(
                "SELECT vecteur FROM empreintes_vocales WHERE personne = ? AND moteur = ?",
                (self.personne, self._nom_du_moteur()),
            ).fetchall()
        if not lignes:
            self._centroide = []
            return self._centroide
        vecteurs = [_depuis_blob(ligne["vecteur"]) for ligne in lignes]
        taille = min(len(v) for v in vecteurs)
        somme = [sum(v[i] for v in vecteurs) for i in range(taille)]
        norme = sum(valeur * valeur for valeur in somme) ** 0.5
        self._centroide = [valeur / norme for valeur in somme] if norme else []
        return self._centroide

    def _retenir(self, signature: list[float]) -> None:
        """Ajoute une signature reconnue aux échantillons. Ne lève jamais."""
        try:
            with self._verrou:
                self._db.execute(
                    """INSERT INTO empreintes_vocales
                       (personne, moteur, dimension, vecteur, origine, horodatage)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (self.personne, self._nom_du_moteur(), len(signature),
                     _en_blob(signature), "usage", _maintenant()),
                )
                self._elaguer()
                self._db.commit()
                self._centroide = None
                self._compte = None
        except sqlite3.Error:  # pragma: no cover - suivre ne casse pas un tour
            logger.exception("Suivi de la voix impossible.")

    def _elaguer(self) -> None:
        """Garde les plus récents, mais jamais au détriment de l'enrôlement.

        Les échantillons d'enrôlement sont ceux que vous avez donnés
        volontairement : ils restent, quoi qu'il arrive. Ce sont ceux ramassés
        à l'usage qui partent en premier — verrou déjà tenu.
        """
        trop = self._db.execute(
            "SELECT COUNT(*) FROM empreintes_vocales WHERE personne = ? AND moteur = ?",
            (self.personne, self._nom_du_moteur()),
        ).fetchone()[0] - self.echantillons_max
        if trop <= 0:
            return
        self._db.execute(
            """DELETE FROM empreintes_vocales WHERE id IN (
                   SELECT id FROM empreintes_vocales
                   WHERE personne = ? AND moteur = ? AND origine = 'usage'
                   ORDER BY id LIMIT ?)""",
            (self.personne, self._nom_du_moteur(), trop),
        )


def _en_blob(vecteur: list[float]) -> bytes:
    return array.array("f", vecteur).tobytes()


def _depuis_blob(blob: bytes) -> list[float]:
    vecteur = array.array("f")
    vecteur.frombytes(blob)
    return list(vecteur)


def _produit_scalaire(gauche: list[float], droite: list[float]) -> float:
    taille = min(len(gauche), len(droite))
    return sum(gauche[i] * droite[i] for i in range(taille))


def _maintenant() -> str:
    return datetime.now().isoformat(timespec="seconds")


def depuis_config(config, moteur: SpeakerEngine | None = None) -> Locuteur | None:
    """Construit la reconnaissance décrite par ``[voix]``.

    Rend toujours un objet quand la section existe, même éteint : une
    compétence doit pouvoir dire comment l'allumer plutôt que de refuser sans
    rien expliquer — même choix que pour le corpus d'éveil.
    """
    section = config.section("voix")
    chemin = (
        config.resolve_path("voix.fichier")
        or config.resolve_path("memoire.fichier")
        or Path.home() / ".lily" / "memoire.sqlite"
    )
    return Locuteur(
        chemin,
        moteur,
        actif=bool(section.get("actif", False)),
        seuil=float(section.get("seuil", 0.62)),
        echantillons_min=int(section.get("echantillons_min", 3)),
        echantillons_max=int(section.get("echantillons_max", 40)),
        apprentissage_continu=bool(section.get("apprentissage_continu", True)),
        marge_d_apprentissage=float(section.get("marge_d_apprentissage", 0.10)),
    )

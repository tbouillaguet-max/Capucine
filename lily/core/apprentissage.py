"""Ce que Lily apprend de vous, sans jamais réentraîner un modèle.

Trois mécanismes ici, tous sans apprentissage automatique : ce sont des
tables, des compteurs et de la comptabilité honnête. C'est délibéré — affiner
un modèle coûterait des heures de GPU et des milliers d'exemples qu'on n'a
pas, pour un gain que ces trois-là obtiennent tout de suite.

* **Le routage.** Les ``examples`` d'un plugin sont écrits par son auteur, qui
  ne sait pas comment *vous* parlez. Quand l'étage déterministe rate et que le
  modèle tranche, on retient la phrase. La fois suivante, l'étage déterministe
  reconnaît seul — plus vite, et sans dépendre de l'humeur d'un 7B.
* **Les corrections.** « Non, je voulais dire le minuteur » est le signal le
  plus riche qui existe : il dit à la fois ce qui était faux et ce qui était
  juste. Une correction vaut cinquante observations passives.
* **Le vocabulaire.** Whisper entend « calcul risque » là où vous dites
  « CalculRisque ». On collecte vos noms propres et on les lui souffle.

Tout vit dans le fichier SQLite de la mémoire. Rien ne sort de la machine.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .logging import get_logger
from .sqlite import regler_la_base
from .text import PhrasePreparee, normalize, strip_accents

logger = get_logger("apprentissage")

SCHEMA = """
CREATE TABLE IF NOT EXISTS routages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    phrase        TEXT NOT NULL,
    normalisee    TEXT NOT NULL,
    outil         TEXT NOT NULL,
    confirmations INTEGER NOT NULL DEFAULT 1,
    dementis      INTEGER NOT NULL DEFAULT 0,
    horodatage    TEXT NOT NULL,
    UNIQUE(normalisee, outil)
);
CREATE INDEX IF NOT EXISTS idx_routages_outil ON routages(outil);
CREATE TABLE IF NOT EXISTS vocabulaire (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mot         TEXT NOT NULL UNIQUE,
    occurrences INTEGER NOT NULL DEFAULT 1,
    source      TEXT NOT NULL,
    horodatage  TEXT NOT NULL
);
"""

# Un nom propre, un identifiant en casse chameau, un mot contenant un chiffre :
# ce que Whisper écorche et qu'aucun dictionnaire français ne contient.
# Volontairement conservateur : on ne retient que ce qu'aucun dictionnaire
# français ne contient. Un simple mot capitalisé serait trop bruyant — et un
# faux positif dans l'amorce est nuisible, il biaise la transcription vers des
# mots que vous ne dites jamais.
_CANDIDATS = re.compile(
    r"(?<![\w-])("
    # CalculRisque, CalculRisque_Mark5, MonProjet2
    r"[A-ZÀ-Ý][a-zà-ÿ]+(?:[A-ZÀ-Ý][a-zà-ÿ]*|_[A-Za-zÀ-ÿ0-9]+|[0-9]+)+"
    r"|[A-ZÀ-Ý]{2,}(?:[0-9_][A-Za-z0-9_]*)?"       # SEC, EBITDA, API, S&P -> SP
    r"|[a-zA-ZÀ-ÿ]+[0-9]+[A-Za-zÀ-ÿ0-9_]*"         # Mark5, qwen2.5 -> qwen2
    r"|[0-9]+[A-ZÀ-Ý][A-Za-zÀ-ÿ0-9_]*"             # 10K, 10Q, 8K (dépôts SEC)
    r")"
)
_MOTS_COURANTS = {
    "je", "tu", "il", "elle", "nous", "vous", "ils", "le", "la", "les", "un",
    "une", "des", "et", "ou", "mais", "donc", "que", "qui", "quoi", "lily",
    "oui", "non", "ok", "bonjour", "bonsoir", "merci",
}


@dataclass
class PhraseApprise:
    phrase: str
    outil: str
    confirmations: int
    dementis: int
    # Préparée à la lecture, comme les exemples d'un plugin : le routeur les
    # compare à chaque tour, et elles ne changent qu'à l'écriture.
    preparee: PhrasePreparee = field(init=False)

    def __post_init__(self) -> None:
        self.preparee = PhrasePreparee.de(self.phrase)

    @property
    def poids(self) -> float:
        """Entre 0,7 et 1,0 : une phrase confirmée plusieurs fois pèse presque
        autant qu'un exemple écrit par l'auteur du plugin, jamais davantage."""
        return min(1.0, 0.7 + 0.1 * (self.confirmations - self.dementis))


@dataclass
class MotAppris:
    mot: str
    occurrences: int
    source: str


class Apprentissage:
    """Le magasin de ce qui s'apprend au fil des tours."""

    def __init__(
        self,
        chemin: str | Path,
        *,
        actif: bool = True,
        routage: bool = True,
        corrections: bool = True,
        vocabulaire: bool = True,
        phrases_par_competence: int = 12,
        mots_vocabulaire: int = 40,
    ) -> None:
        self.chemin = Path(chemin).expanduser()
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self.actif = actif
        self.routage_actif = routage
        self.corrections_actives = corrections
        self.vocabulaire_actif = vocabulaire
        self.phrases_par_competence = phrases_par_competence
        self.mots_vocabulaire = mots_vocabulaire

        self._verrou = threading.RLock()
        self._db = sqlite3.connect(str(self.chemin), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        regler_la_base(self._db)
        with self._verrou:
            self._db.executescript(SCHEMA)
            self._db.commit()
        # Le routeur consulte ce cache à chaque tour : on ne va pas au disque
        # pour chaque phrase entendue.
        self._cache_phrases: dict[str, list[PhraseApprise]] | None = None
        self._cache_amorce: str | None = None

    def fermer(self) -> None:
        with self._verrou:
            self._db.close()

    # Deux caches, deux vies : les phrases apprises ne changent pas quand on
    # retient un nom propre, et le vocabulaire ne change pas quand on apprend
    # un routage. Les confondre faisait relire toute la table des routages au
    # premier nom propre entendu dans un tour.
    def _invalider_routages(self) -> None:
        self._cache_phrases = None

    def _invalider_amorce(self) -> None:
        self._cache_amorce = None

    # -- routage ------------------------------------------------------------
    def apprendre_routage(self, phrase: str, outil: str) -> bool:
        """Retient qu'une phrase mène à un outil. Ne lève jamais."""
        if not (self.actif and self.routage_actif):
            return False
        phrase = phrase.strip()
        normalisee = normalize(phrase)
        if not normalisee or len(normalisee) < 4:
            return False
        try:
            with self._verrou:
                self._db.execute(
                    """INSERT INTO routages (phrase, normalisee, outil, horodatage)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(normalisee, outil) DO UPDATE SET
                           confirmations = confirmations + 1,
                           horodatage = excluded.horodatage""",
                    (phrase, normalisee, outil, _maintenant()),
                )
                self._db.commit()
                self._invalider_routages()
        except sqlite3.Error:
            logger.exception("Apprentissage du routage impossible.")
            return False
        logger.info("Routage appris : « %s » → %s", phrase[:60], outil)
        return True

    def dementir_routage(self, phrase: str, outil: str) -> bool:
        """Enregistre qu'une phrase ne menait pas à cet outil.

        Au-delà d'un démenti de plus que de confirmations, l'association est
        supprimée : une erreur apprise doit pouvoir se désapprendre aussi vite
        qu'elle s'est apprise.
        """
        if not self.actif:
            return False
        normalisee = normalize(phrase)
        with self._verrou:
            self._db.execute(
                "UPDATE routages SET dementis = dementis + 1 WHERE normalisee = ? AND outil = ?",
                (normalisee, outil),
            )
            supprimees = self._db.execute(
                "DELETE FROM routages WHERE normalisee = ? AND outil = ? AND dementis >= confirmations",
                (normalisee, outil),
            ).rowcount
            self._db.commit()
            self._invalider_routages()
        if supprimees:
            logger.info("Routage oublié : « %s » ↛ %s", phrase[:60], outil)
        return True

    def phrases_par_outil(self) -> dict[str, list[PhraseApprise]]:
        """Les phrases retenues, prêtes pour l'étage déterministe du routeur."""
        if not (self.actif and self.routage_actif):
            return {}
        if self._cache_phrases is not None:
            return self._cache_phrases
        with self._verrou:
            lignes = self._db.execute(
                """SELECT phrase, outil, confirmations, dementis FROM routages
                   WHERE confirmations > dementis
                   ORDER BY confirmations - dementis DESC, id DESC"""
            ).fetchall()
        table: dict[str, list[PhraseApprise]] = {}
        for ligne in lignes:
            apprises = table.setdefault(ligne["outil"], [])
            if len(apprises) < self.phrases_par_competence:
                apprises.append(PhraseApprise(
                    ligne["phrase"], ligne["outil"],
                    ligne["confirmations"], ligne["dementis"],
                ))
        self._cache_phrases = table
        return table

    def oublier_outil(self, outil: str) -> int:
        """Efface ce qui a été appris d'un outil disparu."""
        with self._verrou:
            nombre = self._db.execute(
                "DELETE FROM routages WHERE outil = ?", (outil,)
            ).rowcount
            self._db.commit()
            self._invalider_routages()
        return nombre

    # -- vocabulaire --------------------------------------------------------
    def retenir_mot(self, mot: str, source: str = "conversation") -> bool:
        """Compte une occurrence du mot. Renvoie vrai si on ne le connaissait pas.

        La distinction compte : l'appelant veut savoir ce qu'il vient
        d'apprendre, pas ce qu'il vient de recompter.
        """
        mot = mot.strip()
        if not (self.actif and self.vocabulaire_actif) or not self._retenable(mot):
            return False
        with self._verrou:
            # Pas de RETURNING : il demande SQLite 3.35, et la Raspberry Pi de
            # Bullseye en est restée à 3.34. Le verrou rend les deux requêtes
            # aussi atomiques qu'une seule.
            connu = self._db.execute(
                "SELECT 1 FROM vocabulaire WHERE mot = ?", (mot,)
            ).fetchone() is not None
            self._db.execute(
                """INSERT INTO vocabulaire (mot, source, horodatage) VALUES (?, ?, ?)
                   ON CONFLICT(mot) DO UPDATE SET occurrences = occurrences + 1""",
                (mot, source, _maintenant()),
            )
            self._db.commit()
            self._invalider_amorce()
        return not connu

    def moissonner(self, texte: str, source: str = "conversation") -> list[str]:
        """Repère dans un texte ce que Whisper risque d'écorcher.

        Chaque occurrence compte — un mot répété est un mot qui vous tient à
        cœur — mais seuls les mots réellement nouveaux sont renvoyés.

        Toute la moisson tient en **une** transaction. Un `commit` par mot,
        deux fois par tour (la phrase, puis la réponse), c'était autant de
        `fsync` : quelques millisecondes sur un SSD, dix fois plus sur la
        carte SD d'un Raspberry Pi, et une usure d'écriture sans objet.
        """
        if not (self.actif and self.vocabulaire_actif):
            return []
        candidats = [
            mot for mot in (m.strip() for m in _CANDIDATS.findall(texte or ""))
            if self._retenable(mot)
        ]
        if not candidats:
            return []
        maintenant = _maintenant()
        try:
            with self._verrou:
                trous = ",".join("?" for _ in candidats)
                connus = {
                    ligne[0] for ligne in self._db.execute(
                        f"SELECT mot FROM vocabulaire WHERE mot IN ({trous})", candidats
                    )
                }
                self._db.executemany(
                    """INSERT INTO vocabulaire (mot, source, horodatage) VALUES (?, ?, ?)
                       ON CONFLICT(mot) DO UPDATE SET occurrences = occurrences + 1""",
                    [(mot, source, maintenant) for mot in candidats],
                )
                self._db.commit()
                nouveaux = list(dict.fromkeys(m for m in candidats if m not in connus))
                if nouveaux:
                    self._invalider_amorce()
        except sqlite3.Error:
            logger.exception("Moisson de vocabulaire impossible.")
            return []
        return nouveaux

    def _retenable(self, mot: str) -> bool:
        """Ce mot mérite-t-il d'entrer dans le vocabulaire ?"""
        return len(mot) >= 3 and strip_accents(mot).lower() not in _MOTS_COURANTS

    def vocabulaire(self, limite: int = 0) -> list[MotAppris]:
        limite = limite or self.mots_vocabulaire
        with self._verrou:
            lignes = self._db.execute(
                "SELECT mot, occurrences, source FROM vocabulaire "
                "ORDER BY occurrences DESC, id DESC LIMIT ?",
                (limite,),
            ).fetchall()
        return [
            MotAppris(ligne["mot"], ligne["occurrences"], ligne["source"])
            for ligne in lignes
        ]

    def amorce_stt(self, base: str = "") -> str:
        """L'amorce à souffler à Whisper, vocabulaire appris compris.

        Sans elle, « CalculRisque » devient « calcul risque » à chaque fois, et
        la commande qui en dépend rate à chaque fois.
        """
        if not (self.actif and self.vocabulaire_actif):
            return base
        if self._cache_amorce is not None:
            return self._cache_amorce
        mots = [entree.mot for entree in self.vocabulaire()]
        morceaux = [base.strip()] if base.strip() else []
        if mots:
            morceaux.append(", ".join(mots) + ".")
        self._cache_amorce = " ".join(morceaux)
        return self._cache_amorce

    def oublier_mot(self, motif: str) -> int:
        with self._verrou:
            nombre = self._db.execute(
                "DELETE FROM vocabulaire WHERE mot LIKE ?", (f"%{motif}%",)
            ).rowcount
            self._db.commit()
            self._invalider_amorce()
        return nombre

    # -- introspection ------------------------------------------------------
    def statistiques(self) -> dict[str, int]:
        with self._verrou:
            routages = self._db.execute(
                "SELECT COUNT(*) FROM routages WHERE confirmations > dementis"
            ).fetchone()[0]
            outils = self._db.execute(
                "SELECT COUNT(DISTINCT outil) FROM routages WHERE confirmations > dementis"
            ).fetchone()[0]
            mots = self._db.execute("SELECT COUNT(*) FROM vocabulaire").fetchone()[0]
        return {"phrases": routages, "competences": outils, "mots": mots}

    def tout_oublier(self) -> None:
        with self._verrou:
            self._db.execute("DELETE FROM routages")
            self._db.execute("DELETE FROM vocabulaire")
            self._db.commit()
            self._invalider_routages()
            self._invalider_amorce()


def _maintenant() -> str:
    return datetime.now().isoformat(timespec="seconds")


def depuis_config(config) -> Apprentissage | None:
    """Construit l'apprentissage décrit par ``[apprentissage]``."""
    section = config.section("apprentissage")
    if not bool(section.get("active", True)):
        logger.info("Apprentissage désactivé par la configuration.")
        return None
    chemin = (
        config.resolve_path("apprentissage.fichier")
        or config.resolve_path("memoire.fichier")
        or Path.home() / ".lily" / "memoire.sqlite"
    )
    return Apprentissage(
        chemin,
        routage=bool(section.get("routage", True)),
        corrections=bool(section.get("corrections", True)),
        vocabulaire=bool(section.get("vocabulaire", True)),
        phrases_par_competence=int(section.get("phrases_par_competence", 12)),
        mots_vocabulaire=int(section.get("mots_vocabulaire", 40)),
    )

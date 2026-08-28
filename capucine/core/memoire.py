"""La mémoire de Capucine : ce qu'elle garde entre deux démarrages.

Trois choses distinctes, souvent confondues :

* **Le fil courant** — les derniers tours, qui tiennent dans le contexte du
  modèle. C'est ce que gérait déjà ``Conversation``, en mémoire vive.
* **L'historique** — toutes les conversations passées, consultables et
  reprenables. « Reprends notre discussion d'hier sur le backtest. »
* **Les faits durables** — ce qu'elle sait de vous, indépendamment de toute
  conversation. « Je m'appelle Tom », « mon dépôt est dans ~/projets ». Ils
  entrent dans le persona à chaque démarrage.

Le stockage est du SQLite : c'est dans la bibliothèque standard, c'est un seul
fichier, ça résiste à une coupure de courant, et ça sait chercher dans du
texte (FTS5) sans qu'on installe quoi que ce soit. Aucune dépendance ajoutée.

Tout est local, dans ``~/.capucine/memoire.sqlite``. Rien ne sort de la
machine — la règle numéro un du projet vaut aussi pour ce qu'on lui confie.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .logging import get_logger

logger = get_logger("memoire")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    debut       TEXT NOT NULL,
    fin         TEXT,
    titre       TEXT,
    resume      TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    horodatage  TEXT NOT NULL,
    role        TEXT NOT NULL,
    contenu     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS faits (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage  TEXT NOT NULL,
    contenu     TEXT NOT NULL UNIQUE,
    source      TEXT NOT NULL DEFAULT 'utilisateur'
);
"""

# La recherche plein texte est un confort, pas une nécessité : si FTS5 manque,
# on se rabat sur un LIKE. Mieux vaut une recherche moins fine que pas de
# mémoire du tout.
SCHEMA_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    contenu, content='messages', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, contenu) VALUES (new.id, new.contenu);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, contenu)
    VALUES ('delete', old.id, old.contenu);
END;
"""


@dataclass
class Session:
    id: int
    debut: str
    fin: str | None = None
    titre: str | None = None
    resume: str | None = None
    tours: int = 0

    def decrire(self) -> str:
        quand = _lisible(self.debut)
        titre = self.titre or "sans titre"
        return f"#{self.id} — {quand} — {titre} ({self.tours} message{'s' if self.tours > 1 else ''})"


@dataclass
class Extrait:
    session_id: int
    horodatage: str
    role: str
    contenu: str
    titre: str | None = None


@dataclass
class Fait:
    id: int
    contenu: str
    horodatage: str
    source: str = "utilisateur"


class Memoire:
    """Le magasin persistant. Utilisable depuis plusieurs threads."""

    def __init__(self, chemin: str | Path, max_messages_repris: int = 12) -> None:
        self.chemin = Path(chemin).expanduser()
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self.max_messages_repris = max_messages_repris
        self._verrou = threading.Lock()
        # Les plugins et le pipeline appellent depuis des threads différents ;
        # un seul verrou suffit à sérialiser des écritures aussi courtes.
        self._db = sqlite3.connect(str(self.chemin), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._fts = False
        with self._verrou:
            self._db.executescript(SCHEMA)
            try:
                self._db.executescript(SCHEMA_FTS)
                self._fts = True
            except sqlite3.OperationalError as exc:
                logger.info("Recherche plein texte indisponible (%s), repli sur LIKE.", exc)
            self._db.commit()

    def fermer(self) -> None:
        with self._verrou:
            self._db.close()

    # -- sessions -----------------------------------------------------------
    def ouvrir_session(self, titre: str | None = None) -> Session:
        maintenant = _maintenant()
        with self._verrou:
            curseur = self._db.execute(
                "INSERT INTO sessions (debut, titre) VALUES (?, ?)", (maintenant, titre)
            )
            self._db.commit()
        return Session(id=int(curseur.lastrowid), debut=maintenant, titre=titre)

    def fermer_session(self, session_id: int) -> None:
        with self._verrou:
            self._db.execute(
                "UPDATE sessions SET fin = ? WHERE id = ?", (_maintenant(), session_id)
            )
            self._db.commit()

    def sessions(self, limite: int = 10) -> list[Session]:
        with self._verrou:
            lignes = self._db.execute(
                """SELECT s.*, COUNT(m.id) AS tours
                   FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
                   GROUP BY s.id HAVING tours > 0
                   ORDER BY s.id DESC LIMIT ?""",
                (limite,),
            ).fetchall()
        return [_session(ligne) for ligne in lignes]

    def derniere_session(self) -> Session | None:
        sessions = self.sessions(limite=1)
        return sessions[0] if sessions else None

    def session(self, session_id: int) -> Session | None:
        with self._verrou:
            ligne = self._db.execute(
                """SELECT s.*, COUNT(m.id) AS tours
                   FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
                   WHERE s.id = ? GROUP BY s.id""",
                (session_id,),
            ).fetchone()
        return _session(ligne) if ligne else None

    # -- messages -----------------------------------------------------------
    def ajouter_message(self, session_id: int, role: str, contenu: str) -> None:
        if not contenu.strip():
            return
        with self._verrou:
            self._db.execute(
                "INSERT INTO messages (session_id, horodatage, role, contenu) VALUES (?, ?, ?, ?)",
                (session_id, _maintenant(), role, contenu),
            )
            # Le titre d'une session est sa première phrase : pas besoin d'un
            # modèle pour nommer une conversation.
            self._db.execute(
                """UPDATE sessions SET titre = ?
                   WHERE id = ? AND (titre IS NULL OR titre = '') AND ? = 'user'""",
                (contenu.strip()[:70], session_id, role),
            )
            self._db.commit()

    def messages(self, session_id: int, limite: int | None = None) -> list[Extrait]:
        requete = (
            "SELECT session_id, horodatage, role, contenu FROM messages "
            "WHERE session_id = ? ORDER BY id"
        )
        parametres: tuple = (session_id,)
        if limite:
            # Les N derniers, rendus dans l'ordre chronologique. On trie sur
            # l'identifiant et non sur l'horodatage : celui-ci est à la
            # seconde, et deux messages d'un même tour s'y retrouveraient
            # ex æquo, donc dans l'ordre décroissant de la sous-requête.
            requete = (
                "SELECT * FROM (SELECT id, session_id, horodatage, role, contenu "
                "FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id"
            )
            parametres = (session_id, limite)
        with self._verrou:
            lignes = self._db.execute(requete, parametres).fetchall()
        return [
            Extrait(ligne["session_id"], ligne["horodatage"], ligne["role"], ligne["contenu"])
            for ligne in lignes
        ]

    def chercher(self, texte: str, limite: int = 5) -> list[Extrait]:
        """Retrouve des passages de conversations passées."""
        texte = texte.strip()
        if not texte:
            return []
        with self._verrou:
            if self._fts:
                try:
                    lignes = self._db.execute(
                        """SELECT m.session_id, m.horodatage, m.role, m.contenu, s.titre
                           FROM messages_fts f
                           JOIN messages m ON m.id = f.rowid
                           JOIN sessions s ON s.id = m.session_id
                           WHERE messages_fts MATCH ?
                           ORDER BY rank LIMIT ?""",
                        (_requete_fts(texte), limite),
                    ).fetchall()
                except sqlite3.OperationalError:
                    lignes = []
                if lignes:
                    return [_extrait(ligne) for ligne in lignes]
            lignes = self._db.execute(
                """SELECT m.session_id, m.horodatage, m.role, m.contenu, s.titre
                   FROM messages m JOIN sessions s ON s.id = m.session_id
                   WHERE m.contenu LIKE ? ORDER BY m.id DESC LIMIT ?""",
                (f"%{texte}%", limite),
            ).fetchall()
        return [_extrait(ligne) for ligne in lignes]

    # -- faits durables -----------------------------------------------------
    def retenir(self, contenu: str, source: str = "utilisateur") -> bool:
        contenu = contenu.strip()
        if not contenu:
            return False
        with self._verrou:
            curseur = self._db.execute(
                "INSERT OR IGNORE INTO faits (horodatage, contenu, source) VALUES (?, ?, ?)",
                (_maintenant(), contenu, source),
            )
            self._db.commit()
        return curseur.rowcount > 0

    def oublier(self, motif: str) -> int:
        motif = motif.strip()
        if not motif:
            return 0
        with self._verrou:
            curseur = self._db.execute(
                "DELETE FROM faits WHERE contenu LIKE ?", (f"%{motif}%",)
            )
            self._db.commit()
        return curseur.rowcount

    def faits(self, limite: int = 50) -> list[Fait]:
        with self._verrou:
            lignes = self._db.execute(
                "SELECT * FROM faits ORDER BY id DESC LIMIT ?", (limite,)
            ).fetchall()
        return [
            Fait(ligne["id"], ligne["contenu"], ligne["horodatage"], ligne["source"])
            for ligne in lignes
        ]

    def bloc_de_faits(self, limite: int = 30) -> str:
        """Les faits durables, prêts à être ajoutés au persona."""
        faits = self.faits(limite)
        if not faits:
            return ""
        lignes = "\n".join(f"- {fait.contenu}" for fait in reversed(faits))
        return f"Ce que tu sais de ton utilisateur :\n{lignes}"


# --- utilitaires ------------------------------------------------------------

def _maintenant() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _lisible(horodatage: str) -> str:
    try:
        quand = datetime.fromisoformat(horodatage)
    except ValueError:
        return horodatage
    ecart = (datetime.now() - quand).days
    if ecart == 0:
        return f"aujourd'hui à {quand:%H h %M}"
    if ecart == 1:
        return f"hier à {quand:%H h %M}"
    if ecart < 7:
        return f"il y a {ecart} jours"
    return quand.strftime("le %d/%m/%Y")


def _requete_fts(texte: str) -> str:
    """FTS5 a sa propre syntaxe ; on cite chaque mot pour éviter qu'un
    apostrophe ou un tiret ne soit lu comme un opérateur."""
    mots = [mot for mot in texte.replace('"', " ").split() if mot]
    return " OR ".join(f'"{mot}"' for mot in mots) or '""'


def _session(ligne: sqlite3.Row) -> Session:
    return Session(
        id=ligne["id"], debut=ligne["debut"], fin=ligne["fin"],
        titre=ligne["titre"], resume=ligne["resume"],
        # `in ligne` porterait sur les VALEURS de la ligne, pas sur ses
        # colonnes : `.keys()` est ici le seul test correct.
        tours=ligne["tours"] if "tours" in ligne.keys() else 0,  # noqa: SIM118
    )


def _extrait(ligne: sqlite3.Row) -> Extrait:
    return Extrait(
        session_id=ligne["session_id"], horodatage=ligne["horodatage"],
        role=ligne["role"], contenu=ligne["contenu"],
        titre=ligne["titre"] if "titre" in ligne.keys() else None,  # noqa: SIM118
    )


def depuis_config(config) -> Memoire | None:
    """Construit la mémoire décrite par ``[memoire]``, ou rien si désactivée."""
    section = config.section("memoire")
    if not bool(section.get("active", True)):
        logger.info("Mémoire persistante désactivée par la configuration.")
        return None
    chemin = config.resolve_path("memoire.fichier") or Path.home() / ".capucine" / "memoire.sqlite"
    return Memoire(chemin, max_messages_repris=int(section.get("messages_repris", 12)))

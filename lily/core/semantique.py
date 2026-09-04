"""Ce que Lily sait de vos documents et de vos conversations.

C'est le cinquième mécanisme d'apprentissage, et le seul qui apprenne des
*contenus* plutôt que des habitudes. Le principe tient en trois temps :

1. **Découper.** Un rapport de quarante pages ne tient pas dans le contexte
   d'un 7B. On le coupe en fragments d'un paragraphe ou deux.
2. **Vectoriser.** Chaque fragment devient un vecteur où la proximité
   géométrique traduit la proximité de sens. « Combien on a perdu au premier
   trimestre » retrouve « la perte de Q1 s'élève à », sans un mot en commun.
3. **Retrouver, puis répondre.** À la question, on ressort les quelques
   fragments les plus proches et on les donne au modèle local *avec* leur
   provenance. Il répond à partir de ce qu'on lui montre, et on peut dire
   d'où vient la réponse.

Trois principes de conception, qui sont aussi trois refus :

* **Rien ne sort de la machine.** Le vectoriseur est local (Ollama en
  bouclage, ou llama.cpp en processus) — vectoriser un document, c'est en
  envoyer le contenu au moteur.
* **Aucune dépendance nouvelle.** Les vecteurs vivent en BLOB dans le SQLite
  de la mémoire. Pas de base vectorielle à installer, pas de service à tenir
  en vie. La recherche est un balayage : mesurée ici, 171 ms pour 5 000
  fragments en Python pur, 11 ms si numpy est là. Au-delà de quelques
  dizaines de milliers de fragments, il faudra autre chose — et Lily le
  dira plutôt que de ralentir en silence.
* **Sans modèle de plongement, ça marche quand même.** La recherche se rabat
  sur le plein texte (FTS5) : moins fin, jamais absent.
"""

from __future__ import annotations

import array
import hashlib
import queue
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import LilyError
from .interfaces.embeddings import EmbeddingEngine
from .logging import get_logger
from .sqlite import regler_la_base
from .text import split_sentences

logger = get_logger("connaissances")


class IndexPlein(LilyError):
    """L'index a atteint la taille au-delà de laquelle le balayage traîne.

    Une erreur explicite plutôt qu'un ralentissement progressif : Lily
    dit qu'elle est pleine au lieu de mettre trois secondes à répondre sans
    expliquer pourquoi.
    """

SCHEMA = """
CREATE TABLE IF NOT EXISTS fragments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,          -- document | conversation | note
    reference   TEXT NOT NULL,          -- chemin du fichier, ou « conversation 12 »
    ancre       TEXT NOT NULL DEFAULT '',   -- « page 3 », « feuille Bilan », « tour 42 »
    texte       TEXT NOT NULL,
    modele      TEXT NOT NULL,          -- deux modèles = deux espaces incomparables
    dimension   INTEGER NOT NULL,
    vecteur     BLOB,                   -- NULL si indexé sans vectoriseur
    horodatage  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fragments_reference ON fragments(reference);
CREATE INDEX IF NOT EXISTS idx_fragments_modele ON fragments(modele);
CREATE TABLE IF NOT EXISTS documents_indexes (
    reference   TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    empreinte   TEXT NOT NULL,          -- taille + date + début du contenu
    fragments   INTEGER NOT NULL DEFAULT 0,
    horodatage  TEXT NOT NULL
);
"""

# Le plein texte est le filet : sans vectoriseur, la recherche s'y rabat.
SCHEMA_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS fragments_fts USING fts5(
    texte, content='fragments', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS fragments_ai AFTER INSERT ON fragments BEGIN
    INSERT INTO fragments_fts(rowid, texte) VALUES (new.id, new.texte);
END;
CREATE TRIGGER IF NOT EXISTS fragments_ad AFTER DELETE ON fragments BEGIN
    INSERT INTO fragments_fts(fragments_fts, rowid, texte)
    VALUES ('delete', old.id, old.texte);
END;
"""


@dataclass
class Passage:
    """Un fragment retrouvé, avec d'où il vient."""

    texte: str
    reference: str
    ancre: str
    source: str
    score: float

    @property
    def provenance(self) -> str:
        nom = Path(self.reference).name if self.source == "document" else self.reference
        return f"{nom} — {self.ancre}" if self.ancre else nom

    def citer(self) -> str:
        return f"[{self.provenance}]\n{self.texte}"


def decouper(texte: str, taille: int = 900, recouvrement: int = 150) -> list[str]:
    """Coupe un texte en fragments, sur les frontières naturelles d'abord.

    Paragraphes tant qu'ils tiennent, phrases sinon, et coupe franche en
    dernier recours. Le recouvrement évite qu'une réponse tombe pile sur une
    frontière et se retrouve coupée en deux moitiés inutilisables.
    """
    texte = (texte or "").strip()
    if not texte:
        return []
    if len(texte) <= taille:
        return [texte]
    # Un recouvrement aussi large que la taille rendrait la coupe franche
    # sur place : `phrase[taille - recouvrement:]` vaudrait `phrase[0:]` et la
    # boucle ci-dessous ne se terminerait jamais. Les deux valeurs viennent de
    # la configuration, donc on borne ici plutôt que de faire confiance.
    recouvrement = max(0, min(recouvrement, taille // 2))
    pas = max(1, taille - recouvrement)

    morceaux: list[str] = []
    for paragraphe in texte.split("\n\n"):
        paragraphe = paragraphe.strip()
        if not paragraphe:
            continue
        if len(paragraphe) <= taille:
            morceaux.append(paragraphe)
            continue
        for phrase in split_sentences(paragraphe) or [paragraphe]:
            while len(phrase) > taille:
                morceaux.append(phrase[:taille])
                phrase = phrase[pas:]
            if phrase.strip():
                morceaux.append(phrase.strip())

    fragments: list[str] = []
    courant = ""
    for morceau in morceaux:
        if not courant:
            courant = morceau
        elif len(courant) + len(morceau) + 1 <= taille:
            courant = f"{courant}\n{morceau}"
        else:
            fragments.append(courant)
            # Le recouvrement ne s'ajoute que s'il tient : un morceau déjà à
            # la taille limite repart seul, sinon le fragment déborderait de
            # ce que l'appelant a demandé.
            chevauchement = courant[-recouvrement:] if recouvrement else ""
            tient = chevauchement and len(chevauchement) + len(morceau) + 1 <= taille
            courant = f"{chevauchement}\n{morceau}" if tient else morceau
    if courant.strip():
        fragments.append(courant.strip())
    return fragments


class Connaissances:
    """L'index sémantique : découpe, vectorise, retrouve."""

    def __init__(
        self,
        chemin: str | Path,
        moteur: EmbeddingEngine | None = None,
        *,
        taille_fragment: int = 900,
        recouvrement: int = 150,
        passages_rendus: int = 5,
        lot: int = 32,
        indexer_les_conversations: bool = True,
        fragments_max: int = 50000,
    ) -> None:
        self.chemin = Path(chemin).expanduser()
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self.moteur = moteur
        self.taille_fragment = taille_fragment
        self.recouvrement = recouvrement
        self.passages_rendus = passages_rendus
        self.lot = lot
        self.indexer_les_conversations = indexer_les_conversations
        self.fragments_max = fragments_max

        self._verrou = threading.RLock()
        self._db = sqlite3.connect(str(self.chemin), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        regler_la_base(self._db)
        with self._verrou:
            self._db.executescript(SCHEMA)
            try:
                self._db.executescript(SCHEMA_FTS)
                self.plein_texte = True
            except sqlite3.OperationalError:
                logger.info("FTS5 absent : le repli lexical utilisera LIKE.")
                self.plein_texte = False
            self._db.commit()

        # Les tours de conversation partent dans un fil de fond : vectoriser
        # coûte quelques dizaines de millisecondes, et ce n'est pas à
        # l'utilisateur de les attendre après chaque réponse.
        self._file: queue.Queue[tuple[str, str, str, str] | None] = queue.Queue(maxsize=256)
        self._fil: threading.Thread | None = None
        self._ferme = False

    # -- état ---------------------------------------------------------------
    @property
    def vectoriel(self) -> bool:
        """Un vectoriseur est-il réellement disponible ?"""
        return self.moteur is not None

    @property
    def modele(self) -> str:
        return self.moteur.model if self.moteur is not None else "lexical"

    def fermer(self) -> None:
        self._ferme = True
        fil = self._fil
        if fil is not None:
            self._file.put(None)
            fil.join(timeout=5.0)
        with self._verrou:
            self._db.close()
        if self.moteur is not None:
            self.moteur.close()

    # -- indexation ---------------------------------------------------------
    def indexer(
        self,
        reference: str,
        texte: str,
        *,
        source: str = "document",
        ancres: list[str] | None = None,
        empreinte: str = "",
    ) -> int:
        """Indexe un texte sous une référence. Remplace ce qui l'était déjà.

        Renvoie le nombre de fragments écrits. Ne lève pas pour une raison
        d'environnement : un moteur absent indexe sans vecteurs, et la
        recherche lexicale reste possible.
        """
        fragments = decouper(texte, self.taille_fragment, self.recouvrement)
        if not fragments:
            self.oublier(reference)
            return 0
        if self._compter() - self._compter(reference) + len(fragments) > self.fragments_max:
            raise IndexPlein(
                f"L'index atteindrait {self.fragments_max} fragments, sa limite. "
                "Oubliez des documents, ou relevez connaissances.fragments_max."
            )

        vecteurs = self._vectoriser(fragments)
        maintenant = _maintenant()
        modele = self.modele
        dimension = len(vecteurs[0]) if vecteurs and vecteurs[0] else 0
        with self._verrou:
            self._db.execute("DELETE FROM fragments WHERE reference = ?", (reference,))
            self._db.executemany(
                """INSERT INTO fragments
                   (source, reference, ancre, texte, modele, dimension, vecteur, horodatage)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        source, reference,
                        (ancres[index] if ancres and index < len(ancres) else ""),
                        fragment, modele, dimension,
                        _en_blob(vecteurs[index]) if vecteurs else None,
                        maintenant,
                    )
                    for index, fragment in enumerate(fragments)
                ],
            )
            self._db.execute(
                """INSERT INTO documents_indexes (reference, source, empreinte, fragments, horodatage)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(reference) DO UPDATE SET
                       empreinte = excluded.empreinte,
                       fragments = excluded.fragments,
                       horodatage = excluded.horodatage""",
                (reference, source, empreinte, len(fragments), maintenant),
            )
            self._db.commit()
        logger.info("Indexé : %s (%d fragments, %s)", reference, len(fragments), modele)
        return len(fragments)

    @staticmethod
    def empreinte(chemin: str | Path) -> str:
        """Signature d'un fichier, à passer à ``indexer`` et ``deja_indexe``.

        Exposée ici pour qu'un plugin n'ait à importer que ``lily.plugin``.
        """
        return empreinte_de_fichier(Path(chemin))

    def deja_indexe(self, reference: str, empreinte: str) -> bool:
        """Ce document a-t-il déjà été indexé dans cet état exact ?

        C'est ce qui rend « indexe ce dossier » relançable sans tout refaire :
        seuls les fichiers modifiés repassent par le vectoriseur.
        """
        if not empreinte:
            return False
        with self._verrou:
            ligne = self._db.execute(
                "SELECT empreinte FROM documents_indexes WHERE reference = ?", (reference,)
            ).fetchone()
            if ligne is None or ligne["empreinte"] != empreinte:
                return False
            # Changer de modèle de plongement invalide l'index : deux espaces
            # vectoriels différents ne se comparent pas.
            modele = self._db.execute(
                "SELECT modele FROM fragments WHERE reference = ? LIMIT 1", (reference,)
            ).fetchone()
        return modele is None or modele["modele"] == self.modele

    def oublier(self, reference: str) -> int:
        with self._verrou:
            nombre = self._db.execute(
                "DELETE FROM fragments WHERE reference = ?", (reference,)
            ).rowcount
            self._db.execute(
                "DELETE FROM documents_indexes WHERE reference = ?", (reference,)
            )
            self._db.commit()
        return nombre

    def tout_oublier(self) -> None:
        with self._verrou:
            self._db.execute("DELETE FROM fragments")
            self._db.execute("DELETE FROM documents_indexes")
            self._db.commit()

    # -- indexation des conversations, en fond ------------------------------
    def indexer_le_tour(self, reference: str, question: str, reponse: str) -> None:
        """Met un tour de conversation dans la file du fil de fond.

        Jamais bloquant : si la file est pleine, le tour est perdu plutôt que
        de retarder la réponse suivante. Un index de conversation incomplet
        vaut mieux qu'une Lily qui attend.
        """
        if not self.indexer_les_conversations or self._ferme:
            return
        texte = f"{question}\n{reponse}".strip()
        if len(texte) < 40:
            return
        self._demarrer_le_fil()
        try:
            self._file.put_nowait(("conversation", reference, question, reponse))
        except queue.Full:
            logger.debug("File d'indexation pleine : tour non indexé.")

    def _demarrer_le_fil(self) -> None:
        if self._fil is not None and self._fil.is_alive():
            return
        self._fil = threading.Thread(
            target=self._boucle_de_fond, name="lily-index", daemon=True
        )
        self._fil.start()

    def _boucle_de_fond(self) -> None:
        while not self._ferme:
            tache = self._file.get()
            if tache is None:
                return
            _, reference, question, reponse = tache
            try:
                self.indexer(
                    f"{reference} · {datetime.now().strftime('%H:%M')} · {_court(question)}",
                    f"Vous : {question}\nLily : {reponse}",
                    source="conversation",
                )
            except Exception:  # pragma: no cover - indexer ne casse jamais un tour
                logger.exception("Indexation d'un tour impossible.")

    # -- recherche ----------------------------------------------------------
    def chercher(
        self, question: str, limite: int = 0, sources: tuple[str, ...] = ()
    ) -> list[Passage]:
        """Les passages les plus proches de la question.

        Par le sens si un vectoriseur est là, par les mots sinon.
        """
        limite = limite or self.passages_rendus
        question = (question or "").strip()
        if not question:
            return []
        if self.moteur is None:
            return self._chercher_lexical(question, limite, sources)
        try:
            (vecteur,) = self._vectoriser([question])
        except Exception:
            logger.exception("Vectorisation de la question impossible ; repli lexical.")
            return self._chercher_lexical(question, limite, sources)
        if not vecteur:
            return self._chercher_lexical(question, limite, sources)
        return self._chercher_vectoriel(vecteur, limite, sources)

    def _chercher_vectoriel(
        self, vecteur: list[float], limite: int, sources: tuple[str, ...]
    ) -> list[Passage]:
        """Le balayage, en deux temps.

        Classer ne demande que les vecteurs ; les textes, seuls les quelques
        passages retenus en ont besoin. Les rapatrier tous revenait à
        transporter 4,5 Mo de texte pour en garder cinq à 5 000 fragments — et
        quarante-cinq au plafond configuré.
        """
        clause, parametres = _filtre_source(sources)
        with self._verrou:
            lignes = self._db.execute(
                "SELECT id, vecteur FROM fragments "
                f"WHERE vecteur IS NOT NULL AND modele = ?{clause}",
                (self.modele, *parametres),
            ).fetchall()
            if not lignes:
                return []
            scores = _produits_scalaires(vecteur, [ligne["vecteur"] for ligne in lignes])
            classes = sorted(range(len(lignes)), key=lambda index: scores[index], reverse=True)
            retenus = [
                (lignes[index]["id"], scores[index])
                for index in classes[:limite]
                # Un score négatif veut dire « aucun rapport » : mieux vaut rendre
                # trois passages que cinq dont deux hors sujet.
                if scores[index] > 0.05
            ]
            if not retenus:
                return []
            trous = ",".join("?" for _ in retenus)
            details = {
                ligne["id"]: ligne
                for ligne in self._db.execute(
                    f"SELECT id, texte, reference, ancre, source FROM fragments "
                    f"WHERE id IN ({trous})",
                    tuple(identifiant for identifiant, _ in retenus),
                )
            }
        # `IN (…)` ne garantit aucun ordre : on rebâtit depuis le classement.
        return [
            Passage(
                texte=details[identifiant]["texte"],
                reference=details[identifiant]["reference"],
                ancre=details[identifiant]["ancre"],
                source=details[identifiant]["source"],
                score=round(score, 4),
            )
            for identifiant, score in retenus
            if identifiant in details
        ]

    def _chercher_lexical(
        self, question: str, limite: int, sources: tuple[str, ...]
    ) -> list[Passage]:
        clause, parametres = _filtre_source(sources)
        clause_jointe, _ = _filtre_source(sources, prefixe="f.")
        with self._verrou:
            if self.plein_texte:
                try:
                    lignes = self._db.execute(
                        "SELECT f.texte, f.reference, f.ancre, f.source, "
                        "       bm25(fragments_fts) AS rang "
                        "FROM fragments_fts JOIN fragments f ON f.id = fragments_fts.rowid "
                        f"WHERE fragments_fts MATCH ?{clause_jointe} "
                        "ORDER BY rang LIMIT ?",
                        (_requete_fts(question), *parametres, limite),
                    ).fetchall()
                    return [_passage_lexical(ligne) for ligne in lignes]
                except sqlite3.OperationalError:
                    logger.debug("Requête FTS refusée ; repli sur LIKE.")
            lignes = self._db.execute(
                "SELECT texte, reference, ancre, source FROM fragments "
                f"WHERE texte LIKE ?{clause} ORDER BY id DESC LIMIT ?",
                (f"%{question}%", *parametres, limite),
            ).fetchall()
        return [_passage_lexical(ligne) for ligne in lignes]

    def contexte(self, question: str, limite: int = 0, caracteres_max: int = 4000) -> str:
        """Les passages retrouvés, mis en forme pour le modèle.

        Chaque passage porte sa provenance : c'est ce qui permet de répondre
        « d'après le rapport Q1, page 3 » plutôt que d'affirmer sans source.
        """
        blocs: list[str] = []
        total = 0
        for passage in self.chercher(question, limite):
            bloc = passage.citer()
            if total + len(bloc) > caracteres_max:
                break
            blocs.append(bloc)
            total += len(bloc)
        return "\n\n".join(blocs)

    # -- introspection ------------------------------------------------------
    def statistiques(self) -> dict[str, Any]:
        with self._verrou:
            fragments = self._db.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
            documents = self._db.execute(
                "SELECT COUNT(*) FROM documents_indexes WHERE source = 'document'"
            ).fetchone()[0]
            tours = self._db.execute(
                "SELECT COUNT(*) FROM documents_indexes WHERE source = 'conversation'"
            ).fetchone()[0]
        return {
            "fragments": fragments,
            "documents": documents,
            "tours": tours,
            "modele": self.modele,
            "vectoriel": self.vectoriel,
        }

    def references(self, source: str = "document", limite: int = 50) -> list[tuple[str, int]]:
        with self._verrou:
            lignes = self._db.execute(
                "SELECT reference, fragments FROM documents_indexes "
                "WHERE source = ? ORDER BY horodatage DESC LIMIT ?",
                (source, limite),
            ).fetchall()
        return [(ligne["reference"], ligne["fragments"]) for ligne in lignes]

    # -- interne ------------------------------------------------------------
    def _compter(self, reference: str = "") -> int:
        with self._verrou:
            if reference:
                return self._db.execute(
                    "SELECT COUNT(*) FROM fragments WHERE reference = ?", (reference,)
                ).fetchone()[0]
            return self._db.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]

    def _vectoriser(self, textes: list[str]) -> list[list[float]]:
        """Vectorise par lots, et normalise : la similarité devient un produit
        scalaire, et un vecteur venu d'un moteur qui ne normalise pas se
        compare quand même correctement."""
        if self.moteur is None:
            return []
        vecteurs: list[list[float]] = []
        for debut in range(0, len(textes), self.lot):
            vecteurs.extend(self.moteur.encode(textes[debut : debut + self.lot]))
        return [_normaliser(vecteur) for vecteur in vecteurs]


def empreinte_de_fichier(chemin: Path) -> str:
    """Signature bon marché d'un fichier : taille, date, et son premier Ko.

    Pas un hachage complet : ouvrir et lire quarante mégaoctets pour décider
    qu'il n'y a rien à faire serait absurde. Ces trois-là suffisent à repérer
    une modification réelle.
    """
    try:
        etat = chemin.stat()
        with chemin.open("rb") as fichier:
            debut = fichier.read(1024)
    except OSError:
        return ""
    graine = f"{etat.st_size}:{int(etat.st_mtime)}".encode() + debut
    return hashlib.blake2b(graine, digest_size=16).hexdigest()


def _normaliser(vecteur: list[float]) -> list[float]:
    norme = sum(valeur * valeur for valeur in vecteur) ** 0.5
    if norme == 0.0:
        return vecteur
    return [valeur / norme for valeur in vecteur]


def _en_blob(vecteur: list[float]) -> bytes:
    return array.array("f", vecteur).tobytes()


def _produits_scalaires(question: list[float], blobs: list[bytes]) -> list[float]:
    """Le score de chaque fragment. numpy si présent, Python pur sinon.

    Mesuré ici sur 5 000 fragments de 768 dimensions : 11 ms avec numpy,
    171 ms sans. Les deux sont tenables ; la seconde évite une dépendance.
    """
    try:
        import numpy
    except ImportError:
        reference = array.array("f", question)
        scores: list[float] = []
        for blob in blobs:
            vecteur = array.array("f")
            vecteur.frombytes(blob)
            taille = min(len(reference), len(vecteur))
            scores.append(sum(reference[index] * vecteur[index] for index in range(taille)))
        return scores
    dimension = len(question)
    matrice = numpy.frombuffer(b"".join(blobs), dtype=numpy.float32)
    if dimension == 0 or matrice.size % dimension:
        return [0.0] * len(blobs)
    matrice = matrice.reshape(-1, dimension)
    return (matrice @ numpy.asarray(question, dtype=numpy.float32)).tolist()


def _filtre_source(
    sources: tuple[str, ...], prefixe: str = ""
) -> tuple[str, tuple[str, ...]]:
    if not sources:
        return "", ()
    trous = ", ".join("?" for _ in sources)
    return f" AND {prefixe}source IN ({trous})", sources


def _passage_lexical(ligne: sqlite3.Row) -> Passage:
    return Passage(
        texte=ligne["texte"], reference=ligne["reference"], ancre=ligne["ancre"],
        source=ligne["source"], score=0.0,
    )


def _requete_fts(texte: str) -> str:
    mots = [mot for mot in "".join(c if c.isalnum() else " " for c in texte).split() if mot]
    return " OR ".join(f'"{mot}"' for mot in mots) or '""'


def _court(texte: str, taille: int = 40) -> str:
    texte = " ".join((texte or "").split())
    return texte if len(texte) <= taille else texte[: taille - 1] + "…"


def _maintenant() -> str:
    return datetime.now().isoformat(timespec="seconds")


def depuis_config(config, moteur: EmbeddingEngine | None = None) -> Connaissances | None:
    """Construit l'index décrit par ``[connaissances]``.

    Le vectoriseur est passé plutôt que construit ici : la fabrique de moteurs
    sait déjà décider s'il est joignable, et le cœur n'a pas à le refaire.
    """
    section = config.section("connaissances")
    if not bool(section.get("active", True)):
        logger.info("Index des connaissances désactivé par la configuration.")
        return None
    chemin = (
        config.resolve_path("connaissances.fichier")
        or config.resolve_path("memoire.fichier")
        or Path.home() / ".lily" / "memoire.sqlite"
    )
    return Connaissances(
        chemin,
        moteur,
        taille_fragment=int(section.get("taille_fragment", 900)),
        recouvrement=int(section.get("recouvrement", 150)),
        passages_rendus=int(section.get("passages_rendus", 5)),
        indexer_les_conversations=bool(section.get("indexer_les_conversations", True)),
        fragments_max=int(section.get("fragments_max", 50000)),
    )

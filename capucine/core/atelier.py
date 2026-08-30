"""L'atelier : la seule porte par laquelle Capucine touche à vos fichiers.

Ce module existe parce que la commande arrive par la **voix**. Une
transcription est imparfaite, un modèle 7B choisit parfois mal ses arguments,
et « efface le brouillon » peut devenir autre chose. Sans garde-fou, cette
chaîne finit par supprimer un fichier que personne n'a demandé de supprimer.

Le contrat est donc simple et strict :

* **Rien n'est accessible par défaut.** ``[atelier] racines = []`` : la
  capacité est livrée inerte, elle ne s'ouvre que sur décision explicite.
* **Un seul point de résolution.** Tout chemin passe par ``resoudre()``, qui
  développe, résout les liens symboliques, puis vérifie l'appartenance à une
  racine autorisée. Ce qui sort de l'atelier est refusé, point.
* **Des motifs interdits même à l'intérieur.** Un ``.env`` ou une clé SSH
  restent hors de portée quand bien même ils seraient dans une racine.
* **Rien n'est supprimé.** Les fichiers partent à la corbeille, horodatés, et
  toute réécriture laisse une sauvegarde.
"""

from __future__ import annotations

import fnmatch
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .errors import SkillRefused
from .logging import get_logger

logger = get_logger("atelier")

# Ce qui n'a rien à faire entre les mains d'un assistant vocal, même dans un
# dossier autorisé. La liste est complétable, jamais vidée par accident.
MOTIFS_INTERDITS = (
    ".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*",
    "*credential*", "*secret*", "*.kdbx", ".netrc", ".pgpass",
    ".git/config", "*.sqlite-journal",
)
DOSSIERS_INTERDITS = (".ssh", ".gnupg", ".aws", ".config/gcloud")

# Formats que l'atelier écrit en texte UTF-8 détruirait. La liste sert au cas
# où le fichier n'existe pas encore ; quand il existe, c'est son contenu qui
# tranche, ce qui est bien plus sûr qu'une extension.
EXTENSIONS_BINAIRES = frozenset({
    ".xlsx", ".xlsm", ".xlsb", ".xls", ".docx", ".docm", ".doc", ".pptx", ".ppt",
    ".odt", ".ods", ".odp", ".pdf", ".rtf",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz", ".whl", ".jar",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tif", ".tiff",
    ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mkv", ".mov",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".pyd", ".class",
    ".db", ".sqlite", ".sqlite3", ".mdb", ".accdb", ".parquet", ".feather",
    ".onnx", ".gguf", ".pt", ".pth", ".safetensors", ".npy", ".npz",
})
OCTETS_EXAMINES = 8192


class AtelierError(SkillRefused, PermissionError):
    """Chemin hors de l'atelier, ou interdit. Le message dit toujours pourquoi.

    Hérite de ``SkillRefused`` : ce n'est pas un plantage, c'est un refus, et
    Capucine doit le prononcer mot pour mot.
    """


@dataclass
class Atelier:
    """Les dossiers que Capucine a le droit de lire et d'écrire."""

    racines: list[Path] = field(default_factory=list)
    lecture_seule: bool = False
    taille_max_ko: int = 512
    corbeille: Path | None = None
    motifs_interdits: tuple[str, ...] = MOTIFS_INTERDITS
    # Racines demandées mais introuvables. Gardées pour pouvoir le dire au
    # démarrage : un atelier vide sans explication, c'est dix minutes perdues
    # à se demander pourquoi toutes les compétences refusent.
    racines_ignorees: list[str] = field(default_factory=list)

    @property
    def ouvert(self) -> bool:
        return bool(self.racines)

    # -- résolution ---------------------------------------------------------
    def resoudre(self, chemin: str | Path, *, doit_exister: bool = False) -> Path:
        """Transforme un chemin en chemin absolu **vérifié**.

        Lève ``AtelierError`` si le chemin sort des racines autorisées, s'il
        correspond à un motif interdit, ou si l'atelier est fermé.
        """
        if not self.racines:
            raise AtelierError(
                "Aucun dossier de travail n'est ouvert. Renseignez atelier.racines "
                "dans la configuration — par sécurité, la liste est vide au départ."
            )

        candidat = Path(str(chemin)).expanduser()
        if not candidat.is_absolute():
            candidat = self.racines[0] / candidat
        # `resolve` suit les liens symboliques : c'est ce qui empêche un lien
        # posé dans l'atelier de pointer ailleurs.
        candidat = candidat.resolve()

        if not any(_contenu_dans(candidat, racine) for racine in self.racines):
            racines = ", ".join(str(r) for r in self.racines)
            raise AtelierError(
                f"« {chemin} » est hors de l'atelier. Dossiers autorisés : {racines}."
            )

        self._verifier_interdits(candidat)

        if doit_exister and not candidat.exists():
            raise AtelierError(f"« {candidat.name} » n'existe pas.")
        return candidat

    def _verifier_interdits(self, chemin: Path) -> None:
        parties = {p.lower() for p in chemin.parts}
        for interdit in DOSSIERS_INTERDITS:
            if any(morceau.lower() in parties for morceau in Path(interdit).parts):
                raise AtelierError(
                    f"« {chemin.name} » est dans un dossier sensible ({interdit}) : je n'y touche pas."
                )
        nom = chemin.name.lower()
        # Séparateurs normalisés : sous Windows, un motif comme « .git/config »
        # ne rencontrerait jamais « C:\\…\\.git\\config ».
        relatif = str(chemin).lower().replace("\\", "/")
        for motif in self.motifs_interdits:
            if fnmatch.fnmatch(nom, motif) or fnmatch.fnmatch(relatif, f"*{motif}"):
                raise AtelierError(
                    f"« {chemin.name} » correspond à un motif protégé ({motif}) : "
                    "identifiants et clés restent hors de ma portée."
                )

    def verifier_ecriture(self, chemin: Path) -> None:
        if self.lecture_seule:
            raise AtelierError(
                "L'atelier est en lecture seule (atelier.lecture_seule = true)."
            )
        if chemin.is_dir():
            raise AtelierError(f"« {chemin.name} » est un dossier, pas un fichier.")
        self._verifier_texte(chemin)

    def _verifier_texte(self, chemin: Path) -> None:
        """Refuse d'écrire du texte dans un fichier qui n'en est pas.

        L'atelier écrit en UTF-8. Un ``.xlsx`` ou un ``.pdf`` réécrit ainsi
        n'est pas modifié : il est **détruit**. La sauvegarde permettrait de
        revenir en arrière, mais mieux vaut ne pas avoir à s'en servir.

        Quand le fichier existe, c'est son contenu qui tranche — bien plus sûr
        qu'une extension, qui ment souvent. Sinon on se rabat sur l'extension,
        pour ne pas créer un ``.docx`` qui n'en serait pas un.
        """
        if chemin.exists():
            try:
                debut = chemin.open("rb").read(OCTETS_EXAMINES)
            except OSError:
                return
            if not debut:
                return
            binaire = b"\x00" in debut
            if not binaire:
                try:
                    debut.decode("utf-8")
                except UnicodeDecodeError:
                    # Une coupure au milieu d'un caractère multi-octets n'est
                    # pas un fichier binaire : on ne juge que sur le début net.
                    binaire = b"\x00" in debut or not _fin_de_caractere_tronquee(debut)
            if binaire:
                raise AtelierError(
                    f"« {chemin.name} » n'est pas un fichier texte : l'écrire en UTF-8 "
                    "le détruirait. Pour un document Word, Excel, PowerPoint ou PDF, "
                    "passez par les compétences de lecture de documents."
                )
        elif chemin.suffix.lower() in EXTENSIONS_BINAIRES:
            raise AtelierError(
                f"Je ne sais pas créer un « {chemin.suffix} » : ce format n'est pas du "
                "texte. Écrire dedans en UTF-8 produirait un fichier illisible."
            )

    # -- lecture ------------------------------------------------------------
    def lire(self, chemin: str | Path) -> str:
        cible = self.resoudre(chemin, doit_exister=True)
        if cible.is_dir():
            raise AtelierError(f"« {cible.name} » est un dossier.")
        taille_ko = cible.stat().st_size / 1024
        if taille_ko > self.taille_max_ko:
            raise AtelierError(
                f"« {cible.name} » fait {taille_ko:.0f} ko, au-delà de la limite de "
                f"{self.taille_max_ko} ko. Lisez-en un extrait ou relevez "
                "atelier.taille_max_ko."
            )
        return cible.read_text(encoding="utf-8", errors="replace")

    def lister(self, chemin: str | Path = ".", motif: str = "*") -> list[Path]:
        dossier = self.resoudre(chemin, doit_exister=True)
        if not dossier.is_dir():
            return [dossier]
        return sorted(
            entree for entree in dossier.glob(motif)
            if not entree.name.startswith(".") and _sans_erreur(self, entree)
        )

    # -- écriture -----------------------------------------------------------
    def ecrire(self, chemin: str | Path, contenu: str, *, ajouter: bool = False) -> Path:
        """Écrit un fichier, après avoir sauvegardé l'ancien s'il existait."""
        cible = self.resoudre(chemin)
        self.verifier_ecriture(cible)
        if cible.exists() and not ajouter:
            self.sauvegarder(cible)
        cible.parent.mkdir(parents=True, exist_ok=True)
        with cible.open("a" if ajouter else "w", encoding="utf-8") as fichier:
            fichier.write(contenu)
        logger.info("%s %s", "Complété" if ajouter else "Écrit", cible)
        return cible

    def sauvegarder(self, chemin: Path) -> Path:
        """Copie horodatée avant toute réécriture. Jamais de perte silencieuse."""
        horodatage = datetime.now().strftime("%Y%m%d-%H%M%S")
        sauvegarde = chemin.with_name(f"{chemin.name}.{horodatage}.sauvegarde")
        shutil.copy2(chemin, sauvegarde)
        logger.info("Sauvegarde : %s", sauvegarde.name)
        return sauvegarde

    def jeter(self, chemin: str | Path) -> Path:
        """Déplace un fichier à la corbeille. Il n'y a pas de suppression ici.

        Un assistant vocal ne doit pas pouvoir détruire quoi que ce soit de
        façon irrécupérable sur une phrase mal transcrite.
        """
        cible = self.resoudre(chemin, doit_exister=True)
        self.verifier_ecriture(cible)
        corbeille = (self.corbeille or Path.home() / ".capucine" / "corbeille")
        corbeille.mkdir(parents=True, exist_ok=True)
        horodatage = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = corbeille / f"{horodatage}-{cible.name}"
        shutil.move(str(cible), str(destination))
        logger.info("À la corbeille : %s -> %s", cible, destination)
        return destination

    def deplacer(self, source: str | Path, destination: str | Path) -> Path:
        depart = self.resoudre(source, doit_exister=True)
        arrivee = self.resoudre(destination)
        self.verifier_ecriture(depart)
        if arrivee.exists():
            self.sauvegarder(arrivee)
        arrivee.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(depart), str(arrivee))
        return arrivee

    def decrire(self) -> str:
        if not self.racines:
            return "aucun dossier ouvert"
        etat = "lecture seule" if self.lecture_seule else "lecture et écriture"
        return f"{', '.join(str(r) for r in self.racines)} ({etat})"


def _fin_de_caractere_tronquee(debut: bytes) -> bool:
    """Le décodage a-t-il échoué seulement parce qu'on a coupé un caractère ?

    On relit les derniers octets : si tout passe une fois la queue retirée,
    c'était une troncature, pas du binaire.
    """
    for recul in range(1, 5):
        try:
            debut[:-recul].decode("utf-8")
            return True
        except UnicodeDecodeError:
            continue
    return False


def _contenu_dans(chemin: Path, racine: Path) -> bool:
    try:
        return chemin == racine or chemin.is_relative_to(racine)
    except (OSError, ValueError):  # pragma: no cover - chemins exotiques
        return False


def _sans_erreur(atelier: Atelier, chemin: Path) -> bool:
    try:
        atelier._verifier_interdits(chemin)
    except AtelierError:
        return False
    return True


def depuis_config(config) -> Atelier:
    """Construit l'atelier décrit par ``[atelier]``. Vide par défaut."""
    section = config.section("atelier")
    racines: list[Path] = []
    ignorees: list[str] = []
    for brut in section.get("racines", []) or []:
        chemin = Path(str(brut)).expanduser()
        if chemin.is_dir():
            racines.append(chemin.resolve())
        else:
            logger.warning("Racine d'atelier ignorée (dossier absent) : %s", chemin)
            ignorees.append(str(chemin))

    corbeille = section.get("corbeille")
    return Atelier(
        racines=racines,
        lecture_seule=bool(section.get("lecture_seule", False)),
        taille_max_ko=int(section.get("taille_max_ko", 512)),
        corbeille=Path(str(corbeille)).expanduser() if corbeille else None,
        motifs_interdits=tuple(section.get("motifs_interdits", MOTIFS_INTERDITS)),
        racines_ignorees=ignorees,
    )

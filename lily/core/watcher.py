"""Rechargement à chaud : déposer un fichier suffit.

C'est la promesse centrale du projet. Un observateur ``watchdog`` surveille les
dossiers de plugins ; fichier ajouté, modifié ou supprimé, le registre se met à
jour en quelques centaines de millisecondes, sans redémarrage.

Trois précautions valent d'être expliquées, parce qu'aucune ne se devine :

* **Anti-rebond.** Un éditeur ne produit pas un événement par sauvegarde mais
  trois ou quatre : écriture d'un fichier temporaire, renommage atomique,
  changement de droits. Sans regroupement, on rechargerait quatre fois — et la
  première fois sur un fichier tronqué.
* **Empreinte du contenu.** Beaucoup d'outils réécrivent un fichier à
  l'identique (formateurs, `touch`, synchronisation). On compare le hachage
  avant de recharger : pas de changement, pas de rechargement, pas d'annonce
  vocale intempestive.
* **Aucune exception ne remonte.** Un plugin fautif est déjà isolé par le
  registre ; ici on ajoute que le thread de surveillance survit à tout, y
  compris à un dossier qui disparaît sous ses pieds.
"""

from __future__ import annotations

import threading
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .logging import get_logger
from .registry import PluginRegistry

logger = get_logger("surveillance")

DEBOUNCE_MS = 500


class ChangeKind(StrEnum):
    UPSERT = "ajout_ou_modification"
    DELETE = "suppression"


@dataclass
class _EnAttente:
    kind: ChangeKind
    at: float = field(default_factory=time.monotonic)


class PluginWatcher:
    """Surveille les dossiers de plugins et tient le registre à jour.

    La logique d'anti-rebond et d'application est séparée de ``watchdog`` :
    ``notify()`` peut être appelée directement, ce qui rend le comportement
    éprouvable sans dépendre des événements du système de fichiers.
    """

    def __init__(
        self,
        registry: PluginRegistry,
        paths: Iterable[Path] | None = None,
        *,
        debounce_ms: float = DEBOUNCE_MS,
        poll_ms: float = 50.0,
    ) -> None:
        self.registry = registry
        self.paths = [Path(p) for p in (paths if paths is not None else registry.paths)]
        self.debounce_s = debounce_ms / 1000.0
        self.poll_s = poll_ms / 1000.0

        self._en_attente: dict[Path, _EnAttente] = {}
        self._verrou = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer: Any = None
        self.applications = 0   # utile aux tests et au journal

    # -- cycle de vie -------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._thread is not None

    def start(self) -> bool:
        """Démarre la surveillance. ``False`` si watchdog n'est pas installé."""
        if self._thread is not None:
            return True
        try:
            from watchdog.observers import Observer
        except ImportError:
            logger.warning(
                "Rechargement à chaud indisponible : le paquet « watchdog » est absent. "
                "Installez-le avec : pip install watchdog  "
                "(en attendant, la commande /recharge fait le travail à la demande)."
            )
            return False

        gestionnaire = _construire_gestionnaire(self)
        self._observer = Observer()
        surveilles = 0
        for dossier in self.paths:
            if not dossier.is_dir():
                logger.debug("Dossier de plugins absent, non surveillé : %s", dossier)
                continue
            self._observer.schedule(gestionnaire, str(dossier), recursive=False)
            surveilles += 1
        if not surveilles:
            logger.warning("Aucun dossier de plugins à surveiller.")
            self._observer = None
            return False

        self._stop.clear()
        self._observer.start()
        self._thread = threading.Thread(target=self._boucle, name="surveillance", daemon=True)
        self._thread.start()
        logger.info(
            "Rechargement à chaud actif sur %s",
            ", ".join(str(p) for p in self.paths if p.is_dir()),
        )
        return True

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout)
            except Exception:  # pragma: no cover - arrêt au mieux
                logger.debug("Arrêt de l'observateur en erreur.", exc_info=True)
            self._observer = None
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    # -- réception des événements ------------------------------------------
    def notify(self, path: Path | str, kind: ChangeKind) -> None:
        """Enregistre un changement. Le traitement attend la fin de la rafale."""
        chemin = Path(path)
        if not self._concerne(chemin):
            return
        with self._verrou:
            # Une suppression suivie d'une création (renommage atomique) doit
            # se solder par un rechargement, pas par un déchargement.
            self._en_attente[chemin] = _EnAttente(kind)
        logger.debug("Changement noté : %s (%s)", chemin.name, kind.value)

    def _concerne(self, chemin: Path) -> bool:
        if chemin.suffix != ".py" or chemin.name.startswith(("_", ".")):
            return False
        return any(chemin.parent == dossier for dossier in self.paths)

    # -- application --------------------------------------------------------
    def _boucle(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.poll_s)
            try:
                self.flush()
            except Exception:  # pragma: no cover - la surveillance ne meurt pas
                logger.exception("Le rechargement à chaud a échoué.")

    def flush(self, force: bool = False) -> list[Path]:
        """Applique les changements dont la rafale est terminée."""
        maintenant = time.monotonic()
        with self._verrou:
            murs = [
                chemin for chemin, attente in self._en_attente.items()
                if force or maintenant - attente.at >= self.debounce_s
            ]
            lot = {chemin: self._en_attente.pop(chemin) for chemin in murs}
        for chemin, attente in lot.items():
            self._appliquer(chemin, attente.kind)
        return list(lot)

    def _appliquer(self, chemin: Path, kind: ChangeKind) -> None:
        nom = unicodedata.normalize("NFKC", chemin.stem)

        if kind is ChangeKind.DELETE or not chemin.exists():
            if self.registry.unload(nom):
                self.applications += 1
                logger.info("Plugin retiré : %s", nom)
            return

        empreinte_actuelle = _empreinte(chemin)
        connu = self.registry.plugins.get(nom)
        if connu is not None and connu.ok and connu.digest == empreinte_actuelle:
            # Réécriture à l'identique : un formateur, un `touch`, une
            # synchronisation. Recharger n'apporterait rien et annoncerait
            # une compétence « nouvelle » qui ne l'est pas.
            logger.debug("Contenu inchangé, rechargement inutile : %s", chemin.name)
            return

        record = self.registry.reload_file(chemin)
        self.applications += 1
        if record.ok:
            logger.info(
                "Plugin rechargé : %s (%s)", nom, ", ".join(record.skills) or "aucune compétence"
            )
        else:
            logger.error("Plugin %s toujours en échec : %s", nom, record.error)


def _empreinte(chemin: Path) -> str:
    import hashlib

    try:
        return hashlib.sha256(chemin.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def traduire_evenement(watcher: PluginWatcher, event: Any) -> None:
    """Traduit un événement watchdog en appel à ``notify()``.

    Fonction libre plutôt que méthode : elle est ainsi éprouvable avec de
    simples objets factices, sans que watchdog soit installé.
    """
    if getattr(event, "is_directory", False):
        return
    type_evenement = getattr(event, "event_type", "")
    source = getattr(event, "src_path", "")
    destination = getattr(event, "dest_path", "")

    if type_evenement == "moved":
        # Un renommage sort un fichier du dossier et en fait entrer un autre.
        if source:
            watcher.notify(source, ChangeKind.DELETE)
        if destination:
            watcher.notify(destination, ChangeKind.UPSERT)
    elif type_evenement == "deleted":
        watcher.notify(source, ChangeKind.DELETE)
    elif type_evenement in ("created", "modified", "closed"):
        watcher.notify(source, ChangeKind.UPSERT)


def _construire_gestionnaire(watcher: PluginWatcher) -> Any:
    """Fabrique le gestionnaire watchdog, à l'intérieur de ``start()``.

    La classe n'est pas déclarée au niveau du module : watchdog reste
    optionnel, et ``lily.core.watcher`` doit s'importer sans lui.
    """
    from watchdog.events import FileSystemEventHandler

    class GestionnaireWatchdog(FileSystemEventHandler):
        def dispatch(self, event: Any) -> None:
            traduire_evenement(watcher, event)

    return GestionnaireWatchdog()

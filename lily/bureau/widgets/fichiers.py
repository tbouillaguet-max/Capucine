"""Le panneau des fichiers : ce qu'on lui donne, ce qu'elle rend.

Le dossier montré ici est la **racine de l'atelier** — le seul endroit où Lily
a le droit de lire et d'écrire. Ce n'est pas une commodité d'affichage : c'est
le périmètre de sécurité du projet, rendu visible. Ce que vous voyez dans ce
panneau est exactement ce qu'elle peut toucher.

Déposer un fichier sur la fenêtre le copie ici. L'original n'est jamais
déplacé ni modifié : si la copie tourne mal, votre fichier est toujours là où
vous l'aviez laissé.

Ce qui a changé depuis votre dernier regard porte un point vert. Un assistant
qui écrit des fichiers sans le dire oblige à aller vérifier dans l'explorateur ;
autant le montrer.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import theme

# Au-delà, on ne liste plus : un dossier de travail n'est pas une archive, et
# peupler une liste de dix mille entrées fige la fenêtre pour rien.
PLAFOND = 400


class PanneauDesFichiers(QWidget):
    """La liste du dossier de travail, tenue à jour."""

    fichiers_deposes = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panneau")
        # Un QWidget nu ne peint ni fond ni bordure depuis la QSS tant qu'on
        # ne le lui demande pas : sans cet attribut, `QWidget#panneau` reste
        # lettre morte et le panneau flotte sans cadre.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._dossier: Path | None = None
        self._connus: dict[str, float] = {}
        self._nouveaux: set[str] = set()

        entete = QLabel("Dossier de travail", self)
        entete.setObjectName("entete")
        self.chemin = QLabel("—", self)
        self.chemin.setObjectName("chemin")
        self.chemin.setWordWrap(True)

        self.ouvrir = QPushButton("Ouvrir", self)
        self.ouvrir.setObjectName("discret")
        self.ouvrir.setToolTip("Ouvrir le dossier dans l'explorateur")
        self.ouvrir.clicked.connect(self._ouvrir_le_dossier)

        haut = QHBoxLayout()
        haut.setContentsMargins(0, 0, 0, 0)
        haut.addWidget(entete)
        haut.addStretch(1)
        haut.addWidget(self.ouvrir)

        self.liste = QListWidget(self)
        self.liste.setAlternatingRowColors(False)
        self.liste.itemActivated.connect(self._ouvrir_le_fichier)
        self.liste.setToolTip("Double-cliquez pour ouvrir")

        self.pied = QLabel("", self)
        self.pied.setObjectName("legende")

        disposition = QVBoxLayout(self)
        disposition.setContentsMargins(16, 14, 16, 14)
        disposition.setSpacing(10)
        disposition.addLayout(haut)
        disposition.addWidget(self.chemin)
        disposition.addWidget(self.liste, 1)
        disposition.addWidget(self.pied)

        # Les observateurs du système de fichiers manquent des événements —
        # écriture atomique, renommage, réseau. Un rafraîchissement régulier
        # par-dessus coûte trois millisecondes et ne rate rien.
        self._observateur = QFileSystemWatcher(self)
        self._observateur.directoryChanged.connect(self.rafraichir)
        self._battement = QTimer(self)
        self._battement.timeout.connect(self.rafraichir)
        self._battement.setInterval(2500)

    # -- dossier -------------------------------------------------------------
    def ouvrir_le_dossier(self, dossier: Path | None) -> None:
        for ancien in self._observateur.directories():
            self._observateur.removePath(ancien)
        self._dossier = Path(dossier) if dossier else None
        self._connus.clear()
        self._nouveaux.clear()

        if self._dossier is None:
            self.chemin.setText("aucun dossier ouvert")
            self.liste.clear()
            self.pied.setText("")
            self._battement.stop()
            return

        self._dossier.mkdir(parents=True, exist_ok=True)
        self.chemin.setText(str(self._dossier))
        self._observateur.addPath(str(self._dossier))
        self._battement.start()
        self.rafraichir(premier=True)

    @property
    def dossier(self) -> Path | None:
        return self._dossier

    # -- contenu -------------------------------------------------------------
    def rafraichir(self, _chemin: str = "", *, premier: bool = False) -> None:
        if self._dossier is None or not self._dossier.is_dir():
            return
        try:
            entrees = sorted(
                (e for e in self._dossier.iterdir() if not e.name.startswith(".")),
                key=lambda e: e.stat().st_mtime,
                reverse=True,
            )[:PLAFOND]
        except OSError:
            return

        vus: dict[str, float] = {}
        for entree in entrees:
            try:
                date = entree.stat().st_mtime
            except OSError:
                continue
            vus[entree.name] = date
            if not premier and self._connus.get(entree.name) != date:
                self._nouveaux.add(entree.name)
        self._connus = vus
        self._nouveaux &= set(vus)
        self._redessiner(entrees)

    def _redessiner(self, entrees: list[Path]) -> None:
        selection = self.liste.currentItem()
        retenu = selection.data(Qt.UserRole) if selection else None
        self.liste.clear()
        for entree in entrees:
            element = QListWidgetItem(self._libelle(entree))
            element.setData(Qt.UserRole, str(entree))
            if entree.name in self._nouveaux:
                element.setForeground(Qt.GlobalColor.white)
                police = QFont(element.font())
                police.setWeight(QFont.DemiBold)
                element.setFont(police)
                element.setToolTip("nouveau ou modifié")
            self.liste.addItem(element)
            if retenu == str(entree):
                self.liste.setCurrentItem(element)
        nombre = len(entrees)
        neufs = len(self._nouveaux)
        pied = f"{nombre} élément{'s' if nombre > 1 else ''}"
        if neufs:
            pied += f" · {neufs} nouveau{'x' if neufs > 1 else ''}"
        self.pied.setText(pied)

    def _libelle(self, entree: Path) -> str:
        marque = "●  " if entree.name in self._nouveaux else "    "
        if entree.is_dir():
            return f"{marque}{entree.name}/"
        try:
            taille = entree.stat().st_size
            quand = datetime.fromtimestamp(entree.stat().st_mtime).strftime("%H:%M")
        except OSError:
            return f"{marque}{entree.name}"
        return f"{marque}{entree.name}    {_taille(taille)} · {quand}"

    def marquer_comme_vus(self) -> None:
        self._nouveaux.clear()
        self.rafraichir(premier=True)

    # -- dépôt ---------------------------------------------------------------
    def accueillir(self, chemins: list[Path]) -> list[Path]:
        """Copie des fichiers dans le dossier de travail. Ne déplace jamais."""
        if self._dossier is None:
            return []
        copies: list[Path] = []
        for source in chemins:
            source = Path(source)
            if not source.exists():
                continue
            cible = _sans_ecraser(self._dossier / source.name)
            try:
                if source.is_dir():
                    shutil.copytree(source, cible)
                else:
                    shutil.copy2(source, cible)
            except OSError:
                continue
            copies.append(cible)
        if copies:
            self._nouveaux.update(c.name for c in copies)
            self.rafraichir()
        return copies

    # -- ouverture -----------------------------------------------------------
    def _ouvrir_le_dossier(self) -> None:
        if self._dossier is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._dossier)))

    def _ouvrir_le_fichier(self, element: QListWidgetItem) -> None:
        chemin = element.data(Qt.UserRole)
        if chemin:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(chemin)))


def _sans_ecraser(cible: Path) -> Path:
    """Un nom libre : on ne remplace jamais un fichier déposé plus tôt."""
    if not cible.exists():
        return cible
    tronc, suffixe = cible.stem, cible.suffix
    for numero in range(2, 1000):
        candidat = cible.with_name(f"{tronc} ({numero}){suffixe}")
        if not candidat.exists():
            return candidat
    return cible.with_name(f"{tronc}-{datetime.now():%H%M%S}{suffixe}")


def _taille(octets: int) -> str:
    for unite, seuil in (("Mo", 1024 * 1024), ("ko", 1024)):
        if octets >= seuil:
            return f"{octets / seuil:.1f} {unite}"
    return f"{octets} o"


# La feuille de style globale ne connaît pas ce panneau ; ces quelques règles
# lui sont propres et vivent donc avec lui.
STYLE = f"""
QListWidget {{
    color: {theme.TEXTE_DOUX};
    font-size: 12.5px;
}}
QListWidget::item {{ padding: 5px 8px; border-radius: {theme.RAYON_PETIT}px; }}
QListWidget::item:hover {{ background: {theme.SURFACE_HAUTE}; }}
"""

"""La fenêtre de Lily.

Trois zones, et rien d'autre :

* en haut, **ce qu'elle fait** — l'indicateur d'état et les deux
  interrupteurs ;
* au centre, **la conversation**, avec la saisie en bas ;
* à droite, **les fichiers** — ce qu'on lui donne et ce qu'elle rend.

Le dossier de travail mérite un mot. La configuration livre ``atelier.racines``
vide, exprès : une commande arrive par la voix, une transcription est
imparfaite, et le projet refuse d'ouvrir un disque entier sur cette base. Mais
une fenêtre où l'on dépose des fichiers a besoin d'un endroit où les poser.
L'interface en ouvre donc **un seul, qu'elle crée elle-même** — ``Documents/
Lily`` — quand rien n'est configuré. C'est une décision, pas un effet de bord :
elle est dite ici, affichée dans le panneau, et une racine configurée l'emporte
toujours.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.config import Config
from ..core.logging import get_logger
from . import theme
from .pont import EtatDuDemarrage, PontLily
from .widgets.conversation import FilDeConversation
from .widgets.etat import Activite, IndicateurDEtat
from .widgets.fichiers import STYLE as STYLE_FICHIERS
from .widgets.fichiers import PanneauDesFichiers

logger = get_logger("bureau")


def dossier_par_defaut() -> Path:
    """``Documents/Lily`` sous Windows, ``~/Lily`` ailleurs."""
    documents = Path.home() / "Documents"
    racine = documents if documents.is_dir() else Path.home()
    return racine / "Lily"


class Interrupteur(QPushButton):
    """Un bouton à deux états, qui dit ce qu'il fait plutôt que son état.

    « Micro allumé » ne se lit pas : est-ce l'état actuel ou ce qu'on
    obtiendra en cliquant ? Le libellé reste donc fixe et c'est la **couleur**
    qui porte l'état — allumé, il est vert ; éteint, il est sourd.
    """

    bascule = Signal(bool)

    def __init__(self, libelle: str, parent: QWidget | None = None) -> None:
        super().__init__(libelle, parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.toggled.connect(self._repeindre)
        self.toggled.connect(self.bascule)
        self._repeindre(False)

    def _repeindre(self, actif: bool) -> None:
        if actif:
            self.setStyleSheet(
                f"QPushButton {{ background: {theme.VERT_VOILE};"
                f" border: 1px solid {theme.VERT}; color: {theme.VERT_VIF};"
                f" border-radius: {theme.RAYON_PETIT}px; padding: 7px 14px; }}"
                f"QPushButton:hover {{ border-color: {theme.VERT_VIF}; }}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton {{ background: transparent;"
                f" border: 1px solid {theme.BORDURE}; color: {theme.TEXTE_FAIBLE};"
                f" border-radius: {theme.RAYON_PETIT}px; padding: 7px 14px; }}"
                f"QPushButton:hover {{ border-color: {theme.BORDURE_VIVE};"
                f" color: {theme.TEXTE_DOUX}; }}"
            )
        self.setStyleSheet(self.styleSheet() + "QPushButton:disabled { color: #3C4A44; }")


class Saisie(QTextEdit):
    """Le champ de saisie : Entrée envoie, Maj+Entrée va à la ligne."""

    envoyer = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("saisie")
        self.setPlaceholderText("Écrivez à Lily…   (Entrée pour envoyer, Maj+Entrée pour aller à la ligne)")
        self.setAcceptRichText(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFixedHeight(52)
        self.textChanged.connect(self._ajuster)

    def keyPressEvent(self, evenement) -> None:  # noqa: N802 - signature Qt
        entree = evenement.key() in (Qt.Key_Return, Qt.Key_Enter)
        if entree and not (evenement.modifiers() & Qt.ShiftModifier):
            self.envoyer.emit()
            return
        super().keyPressEvent(evenement)

    def _ajuster(self) -> None:
        """Grandit avec le texte, jusqu'à un plafond raisonnable."""
        hauteur = int(self.document().size().height()) + 24
        self.setFixedHeight(max(52, min(hauteur, 150)))


class FenetreLily(QMainWindow):
    """La fenêtre principale."""

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Lily")
        self.resize(1180, 760)
        self.setMinimumSize(QSize(760, 520))
        self.setAcceptDrops(True)

        self.pont = PontLily(config, self)
        self._prete = False

        self._monter_l_interface()
        self._brancher()
        self.pont.demarrer()

    # -- interface -----------------------------------------------------------
    def _monter_l_interface(self) -> None:
        self.indicateur = IndicateurDEtat()
        self.indicateur.regler(Activite.ETEINTE)

        self.titre = QLabel("Lily")
        self.titre.setObjectName("titre")
        self.sous_titre = QLabel("démarrage…")
        self.sous_titre.setObjectName("soustitre")

        identite = QVBoxLayout()
        identite.setContentsMargins(0, 0, 0, 0)
        identite.setSpacing(1)
        identite.addWidget(self.titre)
        identite.addWidget(self.sous_titre)

        self.micro = Interrupteur("Micro")
        self.micro.setToolTip("Écouter, ou non. La fenêtre reste utilisable dans les deux cas.")
        self.voix = Interrupteur("Voix")
        self.voix.setToolTip("Parler à voix haute, ou seulement afficher.")
        for bouton in (self.micro, self.voix):
            bouton.setEnabled(False)

        barre = QWidget()
        barre.setObjectName("barreHaute")
        barre.setAttribute(Qt.WA_StyledBackground, True)
        haut = QHBoxLayout(barre)
        haut.setContentsMargins(22, 14, 22, 14)
        haut.setSpacing(18)
        haut.addLayout(identite)
        haut.addSpacing(14)
        haut.addWidget(self.indicateur)
        haut.addStretch(1)
        haut.addWidget(self.micro)
        haut.addWidget(self.voix)

        self.fil = FilDeConversation()
        self.saisie = Saisie()
        self.bouton_envoyer = QPushButton("Envoyer")
        self.bouton_envoyer.setObjectName("principal")
        self.bouton_envoyer.setEnabled(False)
        self.bouton_envoyer.setCursor(Qt.PointingHandCursor)

        ligne_de_saisie = QHBoxLayout()
        ligne_de_saisie.setContentsMargins(22, 0, 22, 18)
        ligne_de_saisie.setSpacing(10)
        ligne_de_saisie.addWidget(self.saisie, 1)
        ligne_de_saisie.addWidget(self.bouton_envoyer, 0, Qt.AlignBottom)

        centre = QWidget()
        colonne = QVBoxLayout(centre)
        colonne.setContentsMargins(0, 0, 0, 0)
        colonne.setSpacing(10)
        colonne.addWidget(self.fil, 1)
        colonne.addLayout(ligne_de_saisie)

        self.fichiers = PanneauDesFichiers()
        self.fichiers.setStyleSheet(STYLE_FICHIERS)
        self.fichiers.setMinimumWidth(250)

        cote = QWidget()
        marge = QVBoxLayout(cote)
        marge.setContentsMargins(0, 10, 16, 18)
        marge.addWidget(self.fichiers)

        separateur = QSplitter(Qt.Horizontal)
        separateur.addWidget(centre)
        separateur.addWidget(cote)
        separateur.setStretchFactor(0, 1)
        separateur.setStretchFactor(1, 0)
        separateur.setSizes([820, 320])
        separateur.setHandleWidth(6)
        separateur.setChildrenCollapsible(False)

        corps = QWidget()
        pile = QVBoxLayout(corps)
        pile.setContentsMargins(0, 0, 0, 0)
        pile.setSpacing(0)
        pile.addWidget(barre)
        pile.addWidget(separateur, 1)
        self.setCentralWidget(corps)

    def _brancher(self) -> None:
        self.saisie.envoyer.connect(self._envoyer)
        self.bouton_envoyer.clicked.connect(self._envoyer)
        self.micro.bascule.connect(self.pont.regler_micro)
        self.voix.bascule.connect(self.pont.regler_voix)

        self.pont.prete.connect(self._au_demarrage)
        self.pont.echouee.connect(self._en_echec)
        self.pont.etat_change.connect(self._changer_d_etat)
        self.pont.phrase_dite.connect(self.fil.phrase_de_lily)
        self.pont.tour_termine.connect(self._tour_termine)
        self.pont.annonce.connect(lambda texte: self.fil.systeme(texte))
        self.pont.micro_change.connect(self._micro_change)
        self.pont.voix_change.connect(self._voix_change)

        QShortcut(QKeySequence("Ctrl+M"), self, activated=self.micro.click)
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.fil.vider)
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)

    # -- réactions -----------------------------------------------------------
    def _au_demarrage(self, etat: EtatDuDemarrage) -> None:
        self._prete = True
        self.sous_titre.setText(
            f"{etat.llm} · {etat.competences} compétences"
            + (f" · {etat.stt}" if etat.stt != "-" else "")
        )
        self.indicateur.regler(Activite.REPOS)
        self.bouton_envoyer.setEnabled(True)
        self.micro.setEnabled(etat.micro_possible)
        self.voix.setEnabled(etat.voix_possible)
        if not etat.micro_possible:
            self.micro.setToolTip("Aucun micro utilisable sur cette machine.")
        if not etat.voix_possible:
            self.voix.setToolTip("Aucune voix disponible : les réponses seront affichées.")

        dossier = Path(etat.atelier) if etat.atelier else dossier_par_defaut()
        self.fichiers.ouvrir_le_dossier(dossier)
        if not etat.atelier:
            self._ouvrir_l_atelier(dossier)

        self.fil.systeme(
            "Lily est prête. Écrivez-lui, ou allumez le micro."
        )
        for remarque in etat.remarques:
            self.fil.systeme(remarque, alerte=True)
        # La voix est allumée d'emblée si elle existe ; le micro non — après
        # quoi on a passé du temps à lui apprendre à ne pas écouter la pièce,
        # l'ouvrir tout seul au démarrage serait un curieux défaut.
        if etat.voix_possible:
            self.voix.setChecked(True)
        self.saisie.setFocus()

    def _ouvrir_l_atelier(self, dossier: Path) -> None:
        """Ouvre le dossier créé par l'interface comme racine de l'atelier.

        Sans cela, les compétences qui touchent au disque refuseraient une par
        une, et déposer un fichier dans la fenêtre ne servirait à rien.
        """
        assistant = self.pont.assistant
        if assistant is None or assistant.atelier.racines:
            return
        dossier.mkdir(parents=True, exist_ok=True)
        assistant.atelier.racines = [dossier.resolve()]
        assistant.atelier.racines_ignorees = []
        logger.info("Atelier ouvert sur %s", dossier)

    def _en_echec(self, message: str) -> None:
        self.fil.systeme(message, alerte=True)
        if not self._prete:
            self.sous_titre.setText("démarrage impossible")
            self.indicateur.regler(Activite.ETEINTE)

    def _changer_d_etat(self, etat: str) -> None:
        if not self.micro.isChecked() and etat == "idle":
            self.indicateur.regler(Activite.REPOS)
            return
        self.indicateur.depuis_le_pipeline(etat)

    def _tour_termine(self, resultat) -> None:
        self.fil.terminer_le_tour(
            resultat.utterance, resultat.display, _legende(resultat)
        )
        self.fichiers.rafraichir()

    def _micro_change(self, actif: bool) -> None:
        self.micro.blockSignals(True)
        self.micro.setChecked(actif)
        self.micro.blockSignals(False)
        self.indicateur.regler(Activite.ECOUTE if actif else Activite.REPOS)

    def _voix_change(self, active: bool) -> None:
        self.voix.blockSignals(True)
        self.voix.setChecked(active)
        self.voix.blockSignals(False)

    def _envoyer(self) -> None:
        texte = self.saisie.toPlainText().strip()
        if not texte or not self._prete:
            return
        self.saisie.clear()
        self.fil.dire_vous(texte)
        self.pont.envoyer(texte)

    # -- glisser-déposer -----------------------------------------------------
    def dragEnterEvent(self, evenement: QDragEnterEvent) -> None:  # noqa: N802
        if evenement.mimeData().hasUrls():
            evenement.acceptProposedAction()

    def dropEvent(self, evenement: QDropEvent) -> None:  # noqa: N802
        chemins = [
            Path(url.toLocalFile())
            for url in evenement.mimeData().urls()
            if url.isLocalFile()
        ]
        if not chemins:
            return
        copies = self.fichiers.accueillir(chemins)
        if not copies:
            self.fil.systeme("Aucun fichier n'a pu être copié.", alerte=True)
            return
        noms = ", ".join(chemin.name for chemin in copies)
        self.fil.systeme(f"Déposé dans le dossier de travail : {noms}")
        evenement.acceptProposedAction()

    def closeEvent(self, evenement: QCloseEvent) -> None:  # noqa: N802
        self.pont.arreter()
        evenement.accept()


def _legende(resultat) -> str:
    """La ligne sous une réponse : d'où elle vient, et ce qu'elle a coûté."""
    morceaux: list[str] = []
    etages = {
        "regle": "sans modèle",
        "regle_arguments": "outil reconnu, arguments par le modèle",
        "llm": "choisi par le modèle",
        "conversation": "conversation",
        "confirmation": "confirmation",
        "erreur": "échec",
    }
    morceaux.append(etages.get(resultat.tier, resultat.tier))
    if resultat.tool is not None:
        morceaux.append(resultat.tool.name.replace("_", " "))
    total = getattr(resultat.telemetry, "stages", {}).get("reflexion_ms")
    if total:
        morceaux.append(f"{total:.0f} ms")
    return " · ".join(morceaux)


def lancer(config: Config | None = None) -> int:
    """Ouvre la fenêtre et rend le code de sortie."""
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("Lily")
    application.setApplicationDisplayName("Lily")
    application.setStyleSheet(theme.feuille_de_style())
    fenetre = FenetreLily(config)
    fenetre.show()
    return application.exec()

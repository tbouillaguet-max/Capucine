"""Le fil de la conversation.

Deux exigences qui tirent dans le même sens : voir la réponse **arriver**
plutôt qu'apparaître d'un bloc, et savoir **d'où elle vient**.

La première se règle avec ``on_phrase`` : chaque phrase prononcée s'ajoute à
la bulle en cours, comme on la lirait par-dessus l'épaule. La seconde tient en
une ligne de légende sous la réponse — l'étage qui a tranché et la compétence
appelée. Ce n'est pas un détail de développeur : c'est la différence entre
« elle a répondu » et « elle a lancé le script, et c'est réglé ».

Un mot sur l'ordre d'arrivée. Quand vous tapez, votre phrase est connue avant
la réponse. Quand vous parlez, elle ne l'est qu'à la fin du tour — la
transcription arrive après que Lily a commencé à répondre. Plutôt que de
laisser sa réponse s'afficher au-dessus de votre question, la bulle « vous »
est alors **insérée** à sa place dans le fil.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import theme


class Bulle(QFrame):
    """Un message. Le vôtre à droite, le sien à gauche.

    Un ``QLabel`` en mode retour à la ligne ne sait pas dire quelle largeur il
    voudrait : Qt lui en accorde donc le minimum, et une phrase de trente mots
    s'affiche sur une colonne de deux cents pixels. On mesure le texte non
    replié, on plafonne, et on impose le résultat.
    """

    LARGEUR_MAX = 560
    LARGEUR_MIN = 120

    def __init__(self, texte: str, *, de_lily: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.de_lily = de_lily
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)

        self.corps = QLabel(texte, self)
        self.corps.setWordWrap(True)
        self.corps.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.corps.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.legende = QLabel("", self)
        self.legende.setObjectName("legende")
        self.legende.hide()

        disposition = QVBoxLayout(self)
        disposition.setContentsMargins(15, 11, 15, 11)
        disposition.setSpacing(6)
        disposition.addWidget(self.corps)
        disposition.addWidget(self.legende)
        self._peindre()
        self._ajuster_la_largeur()

    def _peindre(self) -> None:
        # `Bulle` et non `QFrame` : QLabel hérite de QFrame, si bien qu'une
        # règle `QFrame { border: … }` dessinait aussi un cadre autour du
        # texte et de la légende, à l'intérieur de la bulle.
        if self.de_lily:
            self.setStyleSheet(
                f"Bulle {{ background: {theme.SURFACE_HAUTE};"
                f" border: 1px solid {theme.BORDURE};"
                f" border-left: 2px solid {theme.VERT};"
                f" border-radius: {theme.RAYON}px; }}"
                f"QLabel {{ background: transparent; border: none;"
                f" color: {theme.TEXTE}; }}"
                f"QLabel#legende {{ color: {theme.TEXTE_FAIBLE}; font-size: 11px; }}"
            )
        else:
            self.setStyleSheet(
                f"Bulle {{ background: {theme.VERT_VOILE};"
                f" border: 1px solid transparent;"
                f" border-radius: {theme.RAYON}px; }}"
                f"QLabel {{ background: transparent; border: none;"
                f" color: {theme.TEXTE}; }}"
            )

    def ajouter(self, phrase: str) -> None:
        """Ajoute une phrase à la bulle, telle qu'elle est prononcée."""
        actuel = self.corps.text()
        self.corps.setText(f"{actuel} {phrase}".strip() if actuel else phrase)
        self._ajuster_la_largeur()

    def remplacer(self, texte: str) -> None:
        self.corps.setText(texte)
        self._ajuster_la_largeur()

    def _ajuster_la_largeur(self) -> None:
        texte = self.corps.text()
        if not texte:
            return
        mesure = QFontMetrics(self.corps.font())
        # +38 : les marges intérieures, plus les bordures.
        ideale = mesure.horizontalAdvance(texte) + 38
        self.setFixedWidth(max(self.LARGEUR_MIN, min(int(ideale), self.LARGEUR_MAX)))

    def annoter(self, texte: str) -> None:
        if not texte:
            return
        self.legende.setText(texte)
        self.legende.show()


class Systeme(QLabel):
    """Une ligne discrète, au milieu : une annonce, un avertissement."""

    def __init__(self, texte: str, *, alerte: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(texte, parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        couleur = theme.ALERTE if alerte else theme.TEXTE_FAIBLE
        self.setStyleSheet(f"color: {couleur}; font-size: 12px; padding: 4px 0;")


class FilDeConversation(QScrollArea):
    """La liste des messages, qui suit toujours le dernier."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._contenu = QWidget()
        self._contenu.setObjectName("filContenu")
        self._pile = QVBoxLayout(self._contenu)
        self._pile.setContentsMargins(22, 20, 22, 20)
        self._pile.setSpacing(12)
        self._pile.addStretch(1)
        self.setWidget(self._contenu)

        self._bulle_ouverte: Bulle | None = None
        self._dernier_texte_envoye = ""

    # -- ajouts --------------------------------------------------------------
    def dire_vous(self, texte: str) -> Bulle:
        self._dernier_texte_envoye = texte.strip()
        return self._poser(Bulle(texte, de_lily=False), a_droite=True)

    def commencer_lily(self) -> Bulle:
        if self._bulle_ouverte is None:
            self._bulle_ouverte = self._poser(Bulle("", de_lily=True), a_droite=False)
        return self._bulle_ouverte

    def phrase_de_lily(self, phrase: str) -> None:
        self.commencer_lily().ajouter(phrase)
        self._suivre()

    def systeme(self, texte: str, *, alerte: bool = False) -> None:
        self._poser(Systeme(texte, alerte=alerte), a_droite=False, pleine_largeur=True)

    def terminer_le_tour(self, utterance: str, display: str, legende: str) -> None:
        """Referme la bulle en cours, et remet votre phrase à sa place.

        Une phrase dite au micro n'est connue qu'à la fin du tour. Sans cette
        insertion, votre question s'afficherait sous la réponse.
        """
        utterance = (utterance or "").strip()
        bulle = self._bulle_ouverte
        if utterance and utterance != self._dernier_texte_envoye:
            vous = Bulle(utterance, de_lily=False)
            index = self._index_de(bulle) if bulle is not None else self._pile.count() - 1
            self._poser(vous, a_droite=True, index=index)
        self._dernier_texte_envoye = ""

        if display:
            bulle = bulle or self.commencer_lily()
            # La réponse complète fait foi : la diffusion phrase à phrase perd
            # la ponctuation des fins de ligne, et un tour sans voix ne diffuse
            # rien du tout.
            bulle.remplacer(display)
        if bulle is not None:
            bulle.annoter(legende)
        self._bulle_ouverte = None
        self._suivre()

    def vider(self) -> None:
        while self._pile.count() > 1:
            element = self._pile.takeAt(0)
            widget = element.widget()
            if widget is not None:
                widget.deleteLater()
        self._bulle_ouverte = None

    # -- interne -------------------------------------------------------------
    def _index_de(self, widget: QWidget | None) -> int:
        """La place d'une bulle dans la pile.

        La pile ne contient pas les bulles mais les **lignes** qui les
        portent — c'est ce qui permet d'aligner à gauche ou à droite. On
        cherche donc la ligne dont la bulle est un enfant, faute de quoi
        l'insertion retomberait toujours à la fin, et la question vocale
        s'afficherait sous la réponse qu'elle a provoquée.
        """
        if widget is None:
            return max(0, self._pile.count() - 1)
        for index in range(self._pile.count()):
            ligne = self._pile.itemAt(index).widget()
            if ligne is widget or (ligne is not None and widget in ligne.children()):
                return index
        return max(0, self._pile.count() - 1)

    def _poser(
        self,
        widget: QWidget,
        *,
        a_droite: bool,
        pleine_largeur: bool = False,
        index: int | None = None,
    ) -> QWidget:
        ligne = QWidget()
        disposition = QHBoxLayout(ligne)
        disposition.setContentsMargins(0, 0, 0, 0)
        disposition.setSpacing(0)
        if pleine_largeur:
            disposition.addWidget(widget, 1)
        elif a_droite:
            disposition.addStretch(1)
            disposition.addWidget(widget)
        else:
            disposition.addWidget(widget)
            disposition.addStretch(1)
        # Avant l'étirement final, qui pousse tout vers le haut.
        place = self._pile.count() - 1 if index is None else index
        self._pile.insertWidget(place, ligne)
        self._suivre()
        return widget

    def _suivre(self) -> None:
        """Colle au bas du fil, une fois la mise en page refaite."""
        QTimer.singleShot(0, self._au_bas)

    def _au_bas(self) -> None:
        barre = self.verticalScrollBar()
        barre.setValue(barre.maximum())

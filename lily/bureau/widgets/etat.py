"""L'indicateur d'état : voir d'un coup d'œil ce qu'elle est en train de faire.

C'est la pièce à laquelle tient tout le reste de l'interface. Une assistante
vocale qui met deux secondes à répondre n'est pas lente : elle est **muette**,
et une seconde de silence sans explication est plus longue que trois secondes
expliquées.

Le parti pris de la palette se joue ici. Les états ne sont pas six couleurs
différentes mais **une seule teinte et six mouvements** :

======================  ==================================================
au repos                un point, immobile
elle écoute             un anneau qui s'ouvre et s'efface, comme une onde
elle transcrit          le même anneau, plus rapide
elle réfléchit          un arc qui tourne
elle agit               deux arcs qui tournent en sens contraire
elle parle              trois barres qui battent
éteinte                 un point gris, immobile
======================  ==================================================

Le mouvement se lit du coin de l'œil, la couleur non — et il ne coûte pas une
couleur de plus à une fenêtre qui en veut peu.
"""

from __future__ import annotations

import math
from enum import StrEnum

from PySide6.QtCore import Property, QPointF, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from .. import theme


class Activite(StrEnum):
    """Ce que l'indicateur sait montrer.

    Volontairement plus grossier que ``State`` du pipeline : l'utilisateur n'a
    pas besoin de distinguer « elle appelle une compétence » de « elle attend
    le disque ». Il a besoin de savoir si ça avance.
    """

    ETEINTE = "éteinte"
    REPOS = "prête"
    ECOUTE = "je vous écoute"
    TRANSCRIT = "j'ai entendu"
    REFLECHIT = "je réfléchis"
    AGIT = "je m'en occupe"
    PARLE = "je parle"


# L'état du pipeline, traduit en quelque chose qui se regarde.
DEPUIS_LE_PIPELINE = {
    "idle": Activite.REPOS,
    "wake": Activite.ECOUTE,
    "listen": Activite.ECOUTE,
    "transcribe": Activite.TRANSCRIT,
    "think": Activite.REFLECHIT,
    "act": Activite.AGIT,
    "speak": Activite.PARLE,
}


class Pastille(QWidget):
    """Le dessin seul : un disque de 28 pixels qui bouge selon l'activité."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(30, 30)
        self._activite = Activite.REPOS
        self._phase = 0.0
        # 40 ms : assez fluide pour ne pas saccader, assez lâche pour ne rien
        # coûter. Une animation d'état ne doit pas peser sur la transcription.
        self._minuteur = QTimer(self)
        self._minuteur.timeout.connect(self._avancer)
        self._minuteur.setInterval(40)

    @property
    def activite(self) -> Activite:
        return self._activite

    def regler(self, activite: Activite) -> None:
        if activite is self._activite:
            return
        self._activite = activite
        immobile = activite in (Activite.REPOS, Activite.ETEINTE)
        if immobile:
            self._minuteur.stop()
            self._phase = 0.0
        elif not self._minuteur.isActive():
            self._minuteur.start()
        self.update()

    def _avancer(self) -> None:
        self._phase = (self._phase + 0.045) % 1.0
        self.update()

    # -- dessin -------------------------------------------------------------
    def paintEvent(self, _evenement) -> None:  # noqa: N802 - signature Qt
        peintre = QPainter(self)
        peintre.setRenderHint(QPainter.Antialiasing)
        centre = QPointF(self.width() / 2, self.height() / 2)

        if self._activite is Activite.ETEINTE:
            self._point(peintre, centre, QColor(theme.TEXTE_FAIBLE), 4.5)
        elif self._activite is Activite.REPOS:
            self._point(peintre, centre, QColor(theme.VERT_SOURD), 4.5)
        elif self._activite in (Activite.ECOUTE, Activite.TRANSCRIT):
            rapide = self._activite is Activite.TRANSCRIT
            self._onde(peintre, centre, vitesse=2.0 if rapide else 1.0)
        elif self._activite is Activite.REFLECHIT:
            self._arcs(peintre, centre, nombre=1)
        elif self._activite is Activite.AGIT:
            self._arcs(peintre, centre, nombre=2)
        elif self._activite is Activite.PARLE:
            self._barres(peintre, centre)

    def _point(self, peintre: QPainter, centre: QPointF, couleur: QColor, rayon: float) -> None:
        peintre.setPen(Qt.NoPen)
        peintre.setBrush(couleur)
        peintre.drawEllipse(centre, rayon, rayon)

    def _onde(self, peintre: QPainter, centre: QPointF, vitesse: float) -> None:
        """Un point plein, et un anneau qui s'en éloigne en s'effaçant."""
        self._point(peintre, centre, QColor(theme.VERT), 4.5)
        progression = (self._phase * vitesse) % 1.0
        rayon = 5.0 + progression * 9.0
        couleur = QColor(theme.VERT)
        couleur.setAlphaF(max(0.0, 0.55 * (1.0 - progression)))
        peintre.setBrush(Qt.NoBrush)
        peintre.setPen(QPen(couleur, 1.8))
        peintre.drawEllipse(centre, rayon, rayon)

    def _arcs(self, peintre: QPainter, centre: QPointF, nombre: int) -> None:
        """Un ou deux arcs qui tournent — la seule chose qui dise « ça avance »."""
        peintre.setBrush(Qt.NoBrush)
        for index in range(nombre):
            rayon = 10.5 - index * 4.0
            sens = 1 if index == 0 else -1
            couleur = QColor(theme.VERT if index == 0 else theme.VERT_SOURD)
            peintre.setPen(QPen(couleur, 2.2, Qt.SolidLine, Qt.RoundCap))
            depart = int(sens * self._phase * 360 * 16) + index * 2880
            boite = QRectF(centre.x() - rayon, centre.y() - rayon, rayon * 2, rayon * 2)
            peintre.drawArc(boite, depart, 100 * 16)

    def _barres(self, peintre: QPainter, centre: QPointF) -> None:
        """Trois barres qui battent, décalées : la silhouette de la parole."""
        peintre.setPen(Qt.NoPen)
        peintre.setBrush(QColor(theme.VERT_VIF))
        for index, decalage in enumerate((0.0, 0.33, 0.66)):
            amplitude = abs(math.sin((self._phase + decalage) * math.pi * 2))
            hauteur = 5.0 + amplitude * 13.0
            x = centre.x() - 7.0 + index * 6.0
            chemin = QPainterPath()
            chemin.addRoundedRect(
                QRectF(x - 1.6, centre.y() - hauteur / 2, 3.2, hauteur), 1.6, 1.6
            )
            peintre.fillPath(chemin, peintre.brush())


class IndicateurDEtat(QWidget):
    """La pastille et son libellé, côte à côte."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pastille = Pastille(self)
        self.libelle = QLabel(Activite.REPOS.value, self)
        self.libelle.setObjectName("soustitre")

        disposition = QHBoxLayout(self)
        disposition.setContentsMargins(0, 0, 0, 0)
        disposition.setSpacing(9)
        disposition.addWidget(self.pastille)
        disposition.addWidget(self.libelle)
        disposition.addStretch(1)

        self._fondu = QPropertyAnimation(self, b"opacite_du_libelle", self)
        self._fondu.setDuration(160)

    def regler(self, activite: Activite) -> None:
        if activite is self.pastille.activite:
            return
        self.pastille.regler(activite)
        self.libelle.setText(activite.value)

    def depuis_le_pipeline(self, etat: str) -> None:
        self.regler(DEPUIS_LE_PIPELINE.get(etat, Activite.REPOS))

    # Une propriété Qt, pour que le fondu du libellé soit animable.
    def _lire_opacite(self) -> float:
        return 1.0

    def _ecrire_opacite(self, valeur: float) -> None:
        self.libelle.setStyleSheet(f"color: rgba(140, 157, 149, {valeur});")

    opacite_du_libelle = Property(float, _lire_opacite, _ecrire_opacite)

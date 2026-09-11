"""La palette et la feuille de style de l'application de bureau.

Qt ne donne rien de moderne par défaut : ses widgets arrivent avec des bordures
en relief, des dégradés gris et des boutons qui datent. Tout ce qui suit sert à
les désarmer, puis à reconstruire par-dessus quelque chose de calme.

Trois partis pris, et ils tiennent ensemble :

* **Une seule teinte.** Un vert, décliné en luminance — sourd pour le repos,
  franc pour l'action, vif pour ce qui est en cours. Les états ne se
  distinguent donc pas par la couleur mais par le **mouvement** : un point
  immobile, un anneau qui respire, un arc qui tourne, des barres qui battent.
  C'est ce qui permet de voir d'un coup d'œil si elle réfléchit ou si elle
  parle, sans transformer la fenêtre en sapin de Noël.
* **Des gris qui tirent vers le vert.** Un gris neutre à côté d'un accent
  coloré a toujours l'air d'un gris qu'on a oublié de choisir. Les neutres
  d'ici penchent de quelques degrés vers l'accent.
* **Une seule couleur qui n'est pas verte**, l'ambre des erreurs. Elle est rare
  exprès : si elle apparaît, c'est qu'il faut la lire.
"""

from __future__ import annotations

# --- neutres, légèrement verdis -------------------------------------------
FOND = "#0E1311"
SURFACE = "#151C19"
SURFACE_HAUTE = "#1B2420"
SURFACE_CREUSE = "#0A0E0D"
BORDURE = "#232E29"
BORDURE_VIVE = "#33443C"

# --- texte -----------------------------------------------------------------
TEXTE = "#E6EDE9"
TEXTE_DOUX = "#8C9D95"
TEXTE_FAIBLE = "#5E6D67"

# --- la teinte, en trois luminances ---------------------------------------
VERT = "#48B87E"
VERT_VIF = "#7FE0AB"
VERT_SOURD = "#2A6F4C"
VERT_VOILE = "rgba(72, 184, 126, 0.12)"

# --- la seule couleur qui n'est pas verte ---------------------------------
ALERTE = "#C58A4B"

# Segoe UI Variable est le visage de Windows 11 ; les autres sont des replis,
# pour que la fenêtre reste correcte quand on la développe ailleurs.
POLICE = '"Segoe UI Variable Display", "Segoe UI", "Inter", "DejaVu Sans", sans-serif'
POLICE_MONO = '"Cascadia Code", "Consolas", "JetBrains Mono", "DejaVu Sans Mono", monospace'

RAYON = 10
RAYON_PETIT = 6


def feuille_de_style() -> str:
    """La QSS complète de l'application."""
    return f"""
/* --- fond général ------------------------------------------------------ */
QWidget {{
    background: {FOND};
    color: {TEXTE};
    font-family: {POLICE};
    font-size: 14px;
}}
/* Sans cela, chaque QLabel repeint le fond GÉNÉRAL par-dessus la surface qui
   le porte : des rectangles noirs apparaissent dans la barre haute et dans
   les bulles. C'est le piège le plus courant d'une QSS un peu ambitieuse. */
QLabel {{ background: transparent; }}
QWidget#panneau {{
    background: {SURFACE};
    border: 1px solid {BORDURE};
    border-radius: {RAYON}px;
}}
QWidget#barreHaute {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDURE};
}}

/* --- titres et étiquettes ---------------------------------------------- */
QLabel#titre {{
    font-size: 19px;
    font-weight: 600;
    letter-spacing: 0.2px;
}}
QLabel#soustitre, QLabel#legende {{
    color: {TEXTE_DOUX};
    font-size: 12px;
}}
QLabel#entete {{
    color: {TEXTE_FAIBLE};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.1px;
    text-transform: uppercase;
}}
QLabel#chemin {{
    color: {TEXTE_DOUX};
    font-family: {POLICE_MONO};
    font-size: 11px;
}}
QLabel#alerte {{
    color: {ALERTE};
    font-size: 12px;
}}

/* --- saisie ------------------------------------------------------------ */
QTextEdit#saisie, QLineEdit {{
    background: {SURFACE_HAUTE};
    border: 1px solid {BORDURE};
    border-radius: {RAYON}px;
    padding: 11px 14px;
    color: {TEXTE};
    selection-background-color: {VERT_SOURD};
    selection-color: {TEXTE};
}}
QTextEdit#saisie:focus, QLineEdit:focus {{
    border: 1px solid {VERT};
}}
QTextEdit#saisie[vide="true"] {{
    color: {TEXTE_FAIBLE};
}}

/* --- boutons ------------------------------------------------------------ */
QPushButton {{
    background: {SURFACE_HAUTE};
    border: 1px solid {BORDURE};
    border-radius: {RAYON_PETIT}px;
    padding: 8px 16px;
    color: {TEXTE};
}}
QPushButton:hover {{
    border-color: {BORDURE_VIVE};
    background: #202B26;
}}
QPushButton:pressed {{
    background: {SURFACE_CREUSE};
}}
QPushButton:disabled {{
    color: {TEXTE_FAIBLE};
    border-color: {BORDURE};
}}
QPushButton#principal {{
    background: {VERT};
    border: none;
    color: #06120C;
    font-weight: 600;
}}
QPushButton#principal:hover {{ background: {VERT_VIF}; }}
QPushButton#principal:disabled {{
    background: {VERT_SOURD};
    color: {TEXTE_FAIBLE};
}}
QPushButton#discret {{
    background: transparent;
    border: none;
    color: {TEXTE_DOUX};
    padding: 6px 10px;
}}
QPushButton#discret:hover {{
    color: {VERT_VIF};
    background: {VERT_VOILE};
}}

/* --- listes ------------------------------------------------------------- */
QListWidget {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget::item {{
    padding: 2px 0;
    border: none;
}}
QListWidget::item:selected {{
    background: {VERT_VOILE};
    border-radius: {RAYON_PETIT}px;
}}

/* --- ascenseurs : fins, sans flèches, sans relief ----------------------- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDURE_VIVE};
    border-radius: 3px;
    min-height: 32px;
}}
QScrollBar::handle:vertical:hover {{ background: {VERT_SOURD}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px 4px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDURE_VIVE};
    border-radius: 3px;
    min-width: 32px;
}}

/* --- séparateurs de panneaux -------------------------------------------- */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {BORDURE}; }}

/* --- infobulles ---------------------------------------------------------- */
QToolTip {{
    background: {SURFACE_HAUTE};
    color: {TEXTE};
    border: 1px solid {BORDURE_VIVE};
    border-radius: {RAYON_PETIT}px;
    padding: 6px 9px;
}}

/* --- fenêtres de dialogue ------------------------------------------------ */
QMessageBox, QFileDialog {{ background: {SURFACE}; }}
"""

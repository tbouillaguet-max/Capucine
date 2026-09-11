"""Le lanceur Windows : ce qui devient ``Lily.exe``.

Il ne contient pas Lily. Il la **trouve** et la démarre — c'est tout le sens
d'un exécutable léger : huit méga-octets qui se reconstruisent en dix
secondes, au lieu de deux giga-octets à refaire à chaque modification du
code. Lily reste un dépôt Python ordinaire, qu'on met à jour avec ``git
pull`` sans jamais retoucher l'exécutable.

Il cherche l'installation dans cet ordre :

1. le dossier indiqué dans ``lily.ini``, à côté de l'exécutable ;
2. le dossier de l'exécutable lui-même, et ses parents — le cas normal quand
   on pose ``Lily.exe`` à la racine du dépôt ;
3. à défaut, il **demande**, et retient la réponse.

Aucune dépendance : ``tkinter`` est livré avec Python sous Windows, et sert
ici aux seules boîtes de dialogue. Si le lanceur doit dire quelque chose, il
le dit dans une fenêtre — un exécutable sans console qui échoue en silence
est la pire façon de rater son démarrage.
"""

from __future__ import annotations

import configparser
import subprocess
import sys
from pathlib import Path

FICHIER_DE_REGLAGES = "lily.ini"
POINT_D_ENTREE = "lily_bureau.py"
# Les interpréteurs d'un environnement virtuel, du plus discret au plus
# bavard. `pythonw` n'ouvre pas de console noire derrière la fenêtre.
INTERPRETEURS = (
    Path(".venv") / "Scripts" / "pythonw.exe",
    Path(".venv") / "Scripts" / "python.exe",
    Path("venv") / "Scripts" / "pythonw.exe",
    Path("venv") / "Scripts" / "python.exe",
    Path(".venv") / "bin" / "python",          # pour éprouver ailleurs que sous Windows
    Path("venv") / "bin" / "python",
)


def dossier_de_l_executable() -> Path:
    """Où se trouve l'exécutable — pas le dossier temporaire de PyInstaller."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def dire(titre: str, message: str, *, erreur: bool = True) -> None:
    """Une boîte de dialogue, ou la console si tkinter manque."""
    try:
        import tkinter
        from tkinter import messagebox

        racine = tkinter.Tk()
        racine.withdraw()
        (messagebox.showerror if erreur else messagebox.showinfo)(titre, message)
        racine.destroy()
    except Exception:  # pragma: no cover - tkinter absent d'une build allégée
        print(f"{titre}\n{message}", file=sys.stderr)


def demander_le_dossier() -> Path | None:
    try:
        import tkinter
        from tkinter import filedialog

        racine = tkinter.Tk()
        racine.withdraw()
        choisi = filedialog.askdirectory(title="Où est installée Lily ?")
        racine.destroy()
    except Exception:  # pragma: no cover
        return None
    return Path(choisi) if choisi else None


def ressemble_a_lily(dossier: Path) -> bool:
    return (dossier / POINT_D_ENTREE).is_file() and (dossier / "lily").is_dir()


def lire_les_reglages(a_cote: Path) -> Path | None:
    fichier = a_cote / FICHIER_DE_REGLAGES
    if not fichier.is_file():
        return None
    lecteur = configparser.ConfigParser()
    try:
        lecteur.read(fichier, encoding="utf-8")
        brut = lecteur.get("lily", "dossier", fallback="").strip()
    except configparser.Error:
        return None
    if not brut:
        return None
    dossier = Path(brut).expanduser()
    return dossier if ressemble_a_lily(dossier) else None


def ecrire_les_reglages(a_cote: Path, dossier: Path) -> None:
    lecteur = configparser.ConfigParser()
    lecteur["lily"] = {"dossier": str(dossier)}
    try:
        with (a_cote / FICHIER_DE_REGLAGES).open("w", encoding="utf-8") as fichier:
            lecteur.write(fichier)
    except OSError:  # pragma: no cover - dossier en lecture seule
        pass


def trouver_lily(a_cote: Path) -> Path | None:
    """Le dossier de l'installation, ou ``None`` s'il faut le demander."""
    enregistre = lire_les_reglages(a_cote)
    if enregistre is not None:
        return enregistre
    for candidat in (a_cote, *a_cote.parents):
        if ressemble_a_lily(candidat):
            return candidat
    return None


def trouver_l_interpreteur(dossier: Path) -> Path | None:
    for relatif in INTERPRETEURS:
        chemin = dossier / relatif
        if chemin.is_file():
            return chemin
    return None


def main() -> int:
    a_cote = dossier_de_l_executable()
    dossier = trouver_lily(a_cote)

    if dossier is None:
        choisi = demander_le_dossier()
        if choisi is None:
            return 1
        if not ressemble_a_lily(choisi):
            dire(
                "Lily",
                f"« {choisi} » ne ressemble pas à une installation de Lily :\n"
                f"il faudrait y trouver {POINT_D_ENTREE} et le dossier lily/.",
            )
            return 1
        dossier = choisi
        ecrire_les_reglages(a_cote, dossier)

    interpreteur = trouver_l_interpreteur(dossier)
    if interpreteur is None:
        dire(
            "Lily",
            f"Aucun environnement Python trouvé dans {dossier}.\n\n"
            "Créez-le une fois, depuis ce dossier :\n"
            "    py -3.11 -m venv .venv\n"
            '    .venv\\Scripts\\pip install -e ".[bureau]"\n\n'
            "Puis relancez Lily.",
        )
        return 1

    try:
        # `Popen` et non `run` : le lanceur rend la main tout de suite, la
        # fenêtre vit sa vie. Le code de retour de Lily ne nous regarde pas.
        subprocess.Popen(
            [str(interpreteur), str(dossier / POINT_D_ENTREE), *sys.argv[1:]],
            cwd=str(dossier),
        )
    except OSError as exc:
        dire("Lily", f"Impossible de démarrer Lily :\n{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

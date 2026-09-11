#!/usr/bin/env python3
"""Construit ``Lily.exe`` — à lancer sous Windows.

    py -3.11 -m pip install pyinstaller
    py -3.11 deploy\\construire_exe.py

L'exécutable produit est **léger** : il ne contient pas Lily, il la trouve et
la démarre. Une dizaine de méga-octets, une dizaine de secondes de
construction, et surtout : `git pull` met Lily à jour sans qu'on reconstruise
quoi que ce soit.

Posez ``Lily.exe`` à la racine du dépôt — c'est là qu'il se cherche lui-même —
ou n'importe où, et indiquez-lui le dossier au premier lancement. Il le
retiendra dans un ``lily.ini`` posé à côté de lui.

Pour une icône : déposez ``deploy/lily.ico`` et relancez ce script.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DEPLOY = RACINE / "deploy"
LANCEUR = DEPLOY / "lanceur.py"
ICONE = DEPLOY / "lily.ico"
SORTIE = RACINE / "dist"


def verifier() -> str | None:
    """Ce qui manque pour construire, dit avant de commencer."""
    if sys.platform != "win32":
        return (
            "Ce script produit un exécutable Windows et doit tourner sous Windows.\n"
            "PyInstaller ne pratique pas la compilation croisée : depuis Linux ou\n"
            "macOS, il produirait un binaire pour Linux ou macOS."
        )
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        return "PyInstaller est absent. Installez-le avec :\n    py -3.11 -m pip install pyinstaller"
    if not LANCEUR.is_file():
        return f"Lanceur introuvable : {LANCEUR}"
    return None


def commande() -> list[str]:
    arguments = [
        sys.executable, "-m", "PyInstaller",
        "--name", "Lily",
        # Un seul fichier, pas de console noire derrière la fenêtre.
        "--onefile",
        "--noconsole",
        "--distpath", str(SORTIE),
        "--workpath", str(RACINE / "build" / "pyinstaller"),
        "--specpath", str(RACINE / "build"),
        "--noconfirm",
        # Le lanceur n'a besoin que de la bibliothèque standard. On écarte
        # explicitement ce que PyInstaller croit devoir embarquer : sans cela,
        # un PySide6 présent dans l'environnement gonfle l'exécutable de cent
        # cinquante méga-octets qui ne serviront jamais.
        "--exclude-module", "PySide6",
        "--exclude-module", "numpy",
        "--exclude-module", "PIL",
        "--exclude-module", "lily",
    ]
    if ICONE.is_file():
        arguments += ["--icon", str(ICONE)]
    arguments.append(str(LANCEUR))
    return arguments


def main() -> int:
    probleme = verifier()
    if probleme:
        print(probleme, file=sys.stderr)
        return 1

    print("Construction de Lily.exe…")
    resultat = subprocess.run(commande(), cwd=str(RACINE), check=False)
    if resultat.returncode != 0:
        print("\nLa construction a échoué. La trace de PyInstaller est au-dessus.",
              file=sys.stderr)
        return resultat.returncode

    produit = SORTIE / "Lily.exe"
    if not produit.is_file():
        print(f"\nPyInstaller a fini sans erreur mais {produit} n'existe pas.",
              file=sys.stderr)
        return 1

    # Posé à la racine du dépôt, l'exécutable se trouve tout seul : c'est le
    # cas le plus simple, autant le préparer.
    destination = RACINE / "Lily.exe"
    shutil.copy2(produit, destination)
    taille = destination.stat().st_size / (1024 * 1024)
    print(
        f"\n✓ {destination}  ({taille:.1f} Mo)\n"
        f"  aussi dans {produit}\n\n"
        "Double-cliquez dessus. S'il ne trouve pas l'environnement Python,\n"
        "créez-le une fois depuis ce dossier :\n"
        "    py -3.11 -m venv .venv\n"
        '    .venv\\Scripts\\pip install -e ".[bureau]"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

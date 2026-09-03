"""Accès système : processeur, mémoire, volume.

Ce qu'il montre :

* une **dépendance optionnelle** traitée dans le plugin. ``psutil`` donne de
  meilleurs chiffres, mais son absence ne doit pas rendre la compétence
  inutilisable : on retombe sur ``/proc`` sous Linux. C'est le pendant de la
  règle du cœur — un import manquant *en tête de fichier* écarte le plugin
  avec un message qui nomme le paquet ; ici on choisit de dégrader ;
* du code qui **diffère selon la plateforme**, sans que le cœur en sache rien.
  Le même fichier tourne sur le PC Windows et sur le Raspberry Pi ;
* l'appel de commandes externes avec un délai, pour qu'un outil absent ou
  bloqué ne fige pas Lily.
"""

import platform
import shutil
import subprocess
import time
from pathlib import Path

from lily.plugin import get_logger, skill

CONFIG_DEFAULTS = {
    "delai_commande_s": 3.0,
    "pas_de_volume": 10,
}

SYSTEME = platform.system()


def _psutil():
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def _executer(commande: list[str], delai: float) -> str | None:
    """Lance une commande externe. Retourne ``None` si elle échoue."""
    if not shutil.which(commande[0]):
        return None
    try:
        resultat = subprocess.run(
            commande, capture_output=True, text=True, timeout=delai, check=False
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        get_logger().debug("Commande %s en échec : %s", commande[0], exc)
        return None
    return resultat.stdout if resultat.returncode == 0 else None


# --- processeur et mémoire --------------------------------------------------

def _charge_et_memoire() -> tuple[float | None, float | None]:
    psutil = _psutil()
    if psutil is not None:
        return psutil.cpu_percent(interval=0.3), psutil.virtual_memory().percent

    # Repli sans dépendance, sous Linux : deux lectures de /proc/stat espacées.
    if SYSTEME != "Linux":
        return None, None
    try:
        def _instantane() -> tuple[int, int]:
            champs = [int(v) for v in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
            return sum(champs), champs[3]  # total, inactif

        total1, repos1 = _instantane()
        time.sleep(0.3)
        total2, repos2 = _instantane()
        delta_total = total2 - total1
        charge = 100.0 * (1 - (repos2 - repos1) / delta_total) if delta_total else None

        info = {}
        for ligne in Path("/proc/meminfo").read_text().splitlines():
            cle, _, valeur = ligne.partition(":")
            info[cle] = int(valeur.split()[0])
        totale, disponible = info.get("MemTotal", 0), info.get("MemAvailable", 0)
        memoire = 100.0 * (1 - disponible / totale) if totale else None
        return charge, memoire
    except (OSError, ValueError, IndexError, ZeroDivisionError) as exc:
        get_logger().debug("Lecture de /proc impossible : %s", exc)
        return None, None


@skill(
    description="Donne l'état de la machine : processeur, mémoire, disque.",
    examples=[
        "comment va la machine",
        "quel est l'état du système",
        "combien de mémoire est utilisée",
    ],
)
def etat_systeme() -> dict:
    """Relève la charge processeur, la mémoire et l'espace disque libres.

    Utilise ``psutil`` s'il est installé, sinon ``/proc`` sous Linux.
    """
    charge, memoire = _charge_et_memoire()
    libre_go = shutil.disk_usage(Path.home()).free / 1_000_000_000

    morceaux, journal = [], []
    if charge is not None:
        morceaux.append(f"le processeur est à {charge:.0f} pour cent")
        journal.append(f"cpu={charge:.1f}%")
    if memoire is not None:
        morceaux.append(f"la mémoire à {memoire:.0f} pour cent")
        journal.append(f"ram={memoire:.1f}%")
    morceaux.append(f"il reste {libre_go:.0f} gigaoctets sur le disque")
    journal.append(f"disque_libre={libre_go:.1f}Go")

    if charge is None:
        journal.append("psutil absent")
        get_logger().info(
            "Chiffres partiels : installez psutil pour la charge et la mémoire "
            "(pip install psutil)."
        )
    return {"speak": (", ".join(morceaux) + ".").capitalize(), "display": " ".join(journal)}


# --- volume -----------------------------------------------------------------

def _lire_volume(delai: float) -> int | None:
    if SYSTEME == "Linux":
        sortie = _executer(["pactl", "get-sink-volume", "@DEFAULT_SINK@"], delai)
        if sortie and "%" in sortie:
            return int(sortie.split("%")[0].split()[-1])
        sortie = _executer(["amixer", "get", "Master"], delai)
        if sortie and "[" in sortie:
            return int(sortie.split("[")[1].split("%")[0])
    elif SYSTEME == "Darwin":
        sortie = _executer(["osascript", "-e", "output volume of (get volume settings)"], delai)
        if sortie:
            return int(sortie.strip())
    elif SYSTEME == "Windows":
        sortie = _executer([
            "powershell", "-NoProfile", "-Command",
            "(New-Object -ComObject WScript.Shell) | Out-Null; "
            "[audio]::Volume" ,
        ], delai)
        if sortie and sortie.strip().replace(".", "").isdigit():
            return int(round(float(sortie.strip()) * 100))
    return None


def _ecrire_volume(niveau: int, delai: float) -> bool:
    if SYSTEME == "Linux":
        if _executer(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{niveau}%"], delai) is not None:
            return True
        return _executer(["amixer", "set", "Master", f"{niveau}%"], delai) is not None
    if SYSTEME == "Darwin":
        return _executer(["osascript", "-e", f"set volume output volume {niveau}"], delai) is not None
    if SYSTEME == "Windows":
        # Sans dépendance native, on ne peut qu'agir par pas de deux pour cent
        # via les touches multimédia. C'est grossier mais cela fonctionne
        # partout ; pycaw ferait mieux, au prix d'une dépendance.
        return _volume_windows_par_touches(niveau, delai)
    return False


def _volume_windows_par_touches(cible: int, delai: float) -> bool:
    actuel = _lire_volume(delai)
    if actuel is None:
        actuel = 50
    pas = round((cible - actuel) / 2)
    if pas == 0:
        return True
    touche = 175 if pas > 0 else 174  # volume haut / volume bas
    script = (
        "$w = New-Object -ComObject WScript.Shell; "
        f"1..{abs(pas)} | ForEach-Object {{ $w.SendKeys([char]{touche}) }}"
    )
    return _executer(["powershell", "-NoProfile", "-Command", script], delai) is not None


@skill(
    description="Lit ou règle le volume du haut-parleur, de 0 à 100.",
    examples=["monte le son", "mets le volume à trente", "quel est le volume"],
)
def volume(niveau: int = -1) -> dict:
    """Lit le volume, ou le règle.

    Args:
        niveau: Le volume voulu, de 0 à 100. Laissé à -1, le volume est
            seulement lu.
    """
    from lily.plugin import get_config

    delai = float(get_config("delai_commande_s", 3.0))

    if niveau < 0:
        actuel = _lire_volume(delai)
        if actuel is None:
            return {
                "speak": "Je n'arrive pas à lire le volume sur cette machine.",
                "display": f"lecture du volume impossible ({SYSTEME})",
            }
        return {"speak": f"Le volume est à {actuel}.", "display": f"volume={actuel}"}

    niveau = max(0, min(100, int(niveau)))
    if not _ecrire_volume(niveau, delai):
        return {
            "speak": "Je n'arrive pas à régler le volume sur cette machine.",
            "display": f"réglage du volume impossible ({SYSTEME})",
        }
    return {"speak": f"Volume à {niveau}.", "display": f"volume={niveau}"}

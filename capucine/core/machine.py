"""Ce que Capucine sait de la machine qui l'héberge.

Deux usages :

* choisir un profil sans le demander — un ARM sous Linux est très
  probablement un Raspberry Pi ou un mini-PC, pas la machine principale ;
* **prévenir avant que ça rame**. Whisper « small » sur un Pi à 1 Go de RAM
  ne plante pas : il transcrit en douze secondes, ce qui est pire, parce qu'on
  croit à un bug alors qu'on a simplement demandé l'impossible.

Aucune de ces lectures n'échoue jamais : sur une machine inconnue, on rend ce
qu'on a et on se tait.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .logging import get_logger

logger = get_logger("machine")

# Modèles Whisper et mémoire vive raisonnablement nécessaire, en gigaoctets.
BESOIN_WHISPER = {
    "tiny": 0.6, "base": 0.8, "small": 1.6, "medium": 3.5, "large-v3": 7.0,
}
# Ordre décroissant de qualité, pour proposer un repli.
ORDRE_WHISPER = ["large-v3", "medium", "small", "base", "tiny"]


@dataclass
class Machine:
    """Description matérielle, telle qu'on a pu la lire."""

    systeme: str
    architecture: str
    coeurs: int
    memoire_go: float | None = None
    modele: str | None = None          # « Raspberry Pi 4 Model B Rev 1.4 »
    est_pi: bool = False
    accelerateurs: list[str] = field(default_factory=list)

    @property
    def profil_conseille(self) -> str:
        return "pi" if (self.est_pi or _arm_linux(self.systeme, self.architecture)) else "pc"

    def resume(self) -> str:
        morceaux = [self.modele or f"{self.systeme} {self.architecture}"]
        morceaux.append(f"{self.coeurs} cœur{'s' if self.coeurs > 1 else ''}")
        if self.memoire_go:
            morceaux.append(f"{self.memoire_go:.1f} Go de RAM")
        if self.accelerateurs:
            morceaux.append("+".join(self.accelerateurs))
        return ", ".join(morceaux)


def _arm_linux(systeme: str, architecture: str) -> bool:
    return systeme.lower() == "linux" and architecture.lower().startswith(("arm", "aarch"))


def _memoire_go() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / 1_000_000_000
    except ImportError:
        pass
    try:
        for ligne in Path("/proc/meminfo").read_text().splitlines():
            if ligne.startswith("MemTotal:"):
                return int(ligne.split()[1]) / 1_000_000
    except (OSError, ValueError, IndexError):
        pass
    return None


def _modele() -> tuple[str | None, bool]:
    """Nom du modèle de carte, et s'il s'agit d'un Raspberry Pi."""
    for chemin in (Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model")):
        try:
            nom = chemin.read_bytes().decode("utf-8", "replace").strip("\x00 \n")
            if nom:
                return nom, "raspberry pi" in nom.lower()
        except OSError:
            continue
    try:
        for ligne in Path("/proc/cpuinfo").read_text().splitlines():
            # « Model » avec une majuscule est la convention ARM et donne le
            # nom de la carte. Sur x86, « model : 207 » est un numéro de
            # famille de processeur : ce n'est pas ce qu'on cherche.
            if ligne.startswith("Model") and ":" in ligne:
                nom = ligne.split(":", 1)[1].strip()
                if nom and any(c.isalpha() for c in nom):
                    return nom, "raspberry pi" in nom.lower()
    except OSError:
        pass
    return None, False


def _accelerateurs() -> list[str]:
    trouves: list[str] = []
    if shutil.which("nvidia-smi"):
        trouves.append("cuda")
    try:
        import onnxruntime

        fournisseurs = set(onnxruntime.get_available_providers())
        if "CUDAExecutionProvider" in fournisseurs and "cuda" not in trouves:
            trouves.append("cuda")
        if "DmlExecutionProvider" in fournisseurs:
            trouves.append("directml")
    except ImportError:
        pass
    return trouves


def decrire() -> Machine:
    """Relève ce qu'on peut savoir de la machine. Ne lève jamais."""
    modele, est_pi = _modele()
    return Machine(
        systeme=platform.system(),
        architecture=platform.machine(),
        coeurs=os.cpu_count() or 1,
        memoire_go=_memoire_go(),
        modele=modele,
        est_pi=est_pi,
        accelerateurs=_accelerateurs(),
    )


def conseils(config, machine: Machine | None = None) -> list[str]:
    """Repère les réglages qui vont décevoir sur cette machine.

    On ne corrige rien tout seul : la configuration appartient à
    l'utilisateur. On dit ce qui va se passer et ce qu'on ferait à sa place.
    """
    machine = machine or decrire()
    remarques: list[str] = []

    profil = str(config.get("profile", "pc"))
    conseille = machine.profil_conseille
    if profil != conseille:
        remarques.append(
            f"Profil « {profil} » sur une machine qui ressemble à « {conseille} » "
            f"({machine.resume()}). Essayez --profile {conseille}."
        )

    modele = str(config.get("stt.model", "small"))
    besoin = BESOIN_WHISPER.get(modele)
    if besoin and machine.memoire_go and machine.memoire_go < besoin + 0.5:
        repli = _repli_whisper(machine.memoire_go)
        remarques.append(
            f"Whisper « {modele} » demande environ {besoin:.1f} Go et la machine en a "
            f"{machine.memoire_go:.1f}. Il ne plantera pas, il sera lent. "
            f"Essayez stt.model = \"{repli}\", ou stt.engine = \"vosk\"."
        )

    if machine.coeurs <= 2 and modele not in ("tiny", "base") and not machine.accelerateurs:
        remarques.append(
            f"{machine.coeurs} cœur(s) et Whisper « {modele} » : comptez plusieurs "
            "secondes par phrase. Vosk transcrit en temps réel sur ce genre de machine."
        )

    if str(config.get("stt.device", "auto")) == "cuda" and "cuda" not in machine.accelerateurs:
        remarques.append("stt.device = \"cuda\" mais aucun GPU NVIDIA détecté.")

    if machine.est_pi and str(config.get("barge_in.mode", "voix")) == "voix":
        remarques.append(
            "Barge-in « voix » sur un Raspberry Pi : sans annulation d'écho, un "
            "haut-parleur ouvert fait que Capucine s'interrompt elle-même. "
            "barge_in.mode = \"eveil\" est plus sûr."
        )

    if machine.est_pi and int(config.get("llm.num_ctx", 4096)) > 4096:
        remarques.append(
            f"llm.num_ctx = {config.get('llm.num_ctx')} sur un Pi : la mémoire du "
            "modèle grandit avec le contexte. 2048 suffit largement pour un assistant."
        )
    return remarques


def _repli_whisper(memoire_go: float) -> str:
    for modele in ORDRE_WHISPER:
        if BESOIN_WHISPER[modele] + 0.5 <= memoire_go:
            return modele
    return "tiny"

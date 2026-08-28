"""Exécution d'une compétence dans un sous-processus.

À l'étape 1, j'ai annoncé la limite plutôt que de la masquer : **on ne peut
pas tuer un thread en Python**. Un plugin parti en boucle infinie est
abandonné — Capucine répond et le met en quarantaine — mais son thread
continue de brûler du CPU jusqu'à ce qu'il finisse, s'il finit.

Voici la parade, en option et par compétence ::

    @skill(description="…", isolate=True, timeout=5)
    def analyse_lourde(fichier: str) -> str:
        ...

Le sous-processus est réellement tuable. Le prix est réel aussi, et c'est
pourquoi ce n'est pas le défaut :

* 100 à 300 ms de démarrage à chaque appel — sensible sur Raspberry Pi ;
* arguments et retour doivent être sérialisables (pas de connexion ouverte,
  pas d'objet vivant) ;
* aucun état conservé entre deux appels : un minuteur ne peut pas s'isoler.

**Pourquoi un sous-processus explicite et pas ``multiprocessing``.** Le mode
``spawn`` réimporte le module ``__main__`` du parent dans l'enfant : il casse
dès que ``__main__`` n'est pas un script ordinaire (REPL, carnet, ``-c``), et
il rejouerait le programme s'il manquait un garde ``if __name__``. Le mode
``fork``, lui, est franchement dangereux ici : Capucine tient plusieurs
threads — micro, audio, plugins — et forker un processus multi-thread peut
recopier un verrou déjà pris. On lance donc un interpréteur propre, avec ce
module comme point d'entrée.

Le résultat transite par un fichier temporaire, pas par la sortie standard :
un plugin qui fait ``print()`` ne doit pas pouvoir corrompre la réponse.
"""

from __future__ import annotations

import importlib.util
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .errors import SkillCrashed, SkillTimeout
from .logging import get_logger

logger = get_logger("isolation")

# Marge ajoutée au délai du plugin pour couvrir l'amorçage de l'interpréteur
# enfant. Python démarre en 50 à 150 ms ; trois secondes couvrent largement un
# Raspberry Pi à froid, sans faire attendre l'utilisateur dix secondes pour un
# délai d'une seconde. Réglable par plugins.isolate_startup_s.
DEMARRAGE_S = 3.0


# --- côté enfant -----------------------------------------------------------

def _executer(requete: dict[str, Any]) -> tuple[str, Any]:  # pragma: no cover - autre interpréteur
    for chemin in requete["chemins"]:
        if chemin not in sys.path:
            sys.path.insert(0, chemin)

    from capucine.core.plugin import PluginContext, register_context

    register_context(
        PluginContext(
            plugin=requete["plugin"],
            module=requete["module"],
            config=dict(requete["config"]),
            data_dir=Path(requete["data_dir"]) if requete["data_dir"] else None,
        )
    )

    spec = importlib.util.spec_from_file_location(requete["module"], requete["source"])
    if spec is None or spec.loader is None:
        raise ImportError(f"module illisible : {requete['source']}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[requete["module"]] = module
    spec.loader.exec_module(module)

    fonction = getattr(module, requete["fonction"])
    return "ok", fonction(**requete["arguments"])


def _point_d_entree() -> int:  # pragma: no cover - s'exécute dans l'enfant
    requete = pickle.loads(sys.stdin.buffer.read())
    try:
        reponse = _executer(requete)
    except BaseException as exc:  # noqa: BLE001 - on rapporte tout, on ne meurt pas
        reponse = ("erreur", f"{type(exc).__name__}: {exc}")
    try:
        charge = pickle.dumps(reponse)
    except Exception as exc:
        charge = pickle.dumps((
            "erreur",
            f"le retour de « {requete.get('fonction')} » n'est pas sérialisable : {exc}",
        ))
    Path(requete["resultat"]).write_bytes(charge)
    return 0


# --- côté parent -----------------------------------------------------------

def run_isolated(
    source: Path,
    module_name: str,
    func_name: str,
    kwargs: dict[str, Any],
    *,
    timeout: float | None,
    plugin: str,
    config: dict[str, Any] | None = None,
    data_dir: Path | None = None,
    startup_s: float = DEMARRAGE_S,
) -> Any:
    """Exécute une compétence dans un sous-processus tuable.

    Lève ``SkillTimeout`` si le délai est dépassé — et cette fois le processus
    est réellement arrêté — ou ``SkillCrashed`` en cas d'erreur.
    """
    try:
        pickle.dumps(kwargs)
    except Exception as exc:
        raise SkillCrashed(
            f"les arguments de « {func_name} » ne sont pas sérialisables : {exc}. "
            "Une compétence isolée ne reçoit que des données simples."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="capucine-isole-") as dossier:
        fichier_resultat = Path(dossier) / "resultat.pickle"
        requete = {
            "source": str(source),
            "module": module_name,
            "fonction": func_name,
            "arguments": kwargs,
            "plugin": plugin,
            "config": dict(config or {}),
            "data_dir": str(data_dir) if data_dir else None,
            "chemins": [c for c in sys.path if c],
            "resultat": str(fichier_resultat),
        }

        processus = subprocess.Popen(
            [sys.executable, "-m", "capucine.core.isolation"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        delai = timeout + startup_s if timeout else None
        try:
            _, erreurs = processus.communicate(pickle.dumps(requete), timeout=delai)
        except subprocess.TimeoutExpired:
            # Là où un thread serait abandonné, un processus s'arrête pour de bon.
            logger.warning("Compétence isolée « %s » tuée après %s s.", func_name, timeout)
            processus.kill()
            processus.communicate()
            raise SkillTimeout(
                f"délai de {timeout:g} s dépassé (sous-processus arrêté)"
            ) from None

        if erreurs:
            logger.debug("Sortie d'erreur de « %s » : %s", func_name, erreurs.decode(errors="replace")[:2000])

        if not fichier_resultat.exists():
            raise SkillCrashed(
                f"le sous-processus s'est terminé sans résultat (code {processus.returncode})"
            )
        etat, charge = pickle.loads(fichier_resultat.read_bytes())

    if etat == "ok":
        return charge
    raise SkillCrashed(str(charge))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_point_d_entree())

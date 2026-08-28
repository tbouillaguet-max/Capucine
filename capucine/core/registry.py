"""Découverte, chargement et exécution des plugins.

Règles tenues par ce module :

* **Découverte automatique.** Tout ``.py`` d'un dossier de plugins est importé,
  toute fonction décorée ``@skill`` est enregistrée. Aucun manifeste.
* **Isolation des pannes.** Un plugin qui lève à l'import est écarté avec un
  message clair, les autres continuent. Un skill qui lève ou dépasse son délai
  est neutralisé, et Capucine répond qu'elle n'a pas pu exécuter la commande.
* **Dépendances jamais installées automatiquement.** Un import manquant produit
  un message qui nomme le paquet à installer.
* **Rechargement.** ``load_file`` / ``unload`` / ``reload_file`` sont écrits
  pour l'observateur ``watchdog`` de l'étape 4 ; la table des skills est
  remplacée d'un bloc, jamais mutée pendant qu'un tour l'utilise.

Limite assumée : on ne peut pas tuer un thread en Python. Un plugin parti en
boucle infinie sera *abandonné* — Capucine répond et le met en quarantaine —
mais son thread continuera jusqu'à ce qu'il finisse. La seule parade réelle est
le sous-processus, qui coûte 100 à 300 ms sur Pi et interdit l'état en mémoire
(donc le minuteur). Ce sera une option par skill, pas le défaut.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import logging
import queue
import sys
import threading
import time
import types
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ArgumentError, PluginImportError, SkillCrashed, SkillTimeout
from .logging import get_logger, log_with
from .plugin import (
    SKILL_ATTRIBUTE,
    PluginContext,
    SkillDeclaration,
    SkillSpec,
    build_skill_spec,
    clear_context,
    config_defaults,
    register_context,
)
from .schema import coerce_arguments

logger = get_logger("registre")

PLUGIN_NAMESPACE = "capucine.plugins"
DEFAULT_TIMEOUT = 10.0
QUARANTINE_AFTER = 3


@dataclass
class SkillResult:
    """Ce que le pipeline reçoit après l'exécution d'une compétence."""

    ok: bool
    skill: str
    speak: str = ""
    display: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    needs_confirmation: bool = False
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, skill: str, message: str, duration_ms: float = 0.0) -> SkillResult:
        return cls(ok=False, skill=skill, speak=message, display=message,
                   error=message, duration_ms=duration_ms)


@dataclass
class PluginRecord:
    """État d'un fichier de plugin, chargé ou non."""

    name: str
    path: Path
    module_name: str
    status: str = "charge"           # charge | echec
    error: str | None = None
    missing_package: str | None = None
    skills: list[str] = field(default_factory=list)
    loaded_at: float = field(default_factory=time.time)
    digest: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "charge"


def _module_name_for(path: Path) -> str:
    """Nom de module stable et unicode-sûr.

    Le critère d'acceptation du projet utilise littéralement ``plugins/dés.py``.
    On charge par chemin, donc on maîtrise le nom : normalisation NFKC, et le
    module est déposé dans un espace de noms virtuel ``capucine.plugins``.
    """
    stem = unicodedata.normalize("NFKC", path.stem)
    return f"{PLUGIN_NAMESPACE}.{stem}"


def _ensure_namespace() -> None:
    """Crée l'espace de noms virtuel ``capucine.plugins``.

    Les plugins vivent à la racine du dépôt, pas dans le paquet installé :
    recharger à chaud des fichiers utilisateur à l'intérieur d'un paquet
    ``pip`` est une mauvaise idée. L'espace de noms n'existe donc que
    en mémoire.
    """
    if PLUGIN_NAMESPACE in sys.modules:
        return
    namespace = types.ModuleType(PLUGIN_NAMESPACE)
    namespace.__path__ = []  # type: ignore[attr-defined]
    namespace.__doc__ = "Espace de noms virtuel des plugins chargés à chaud."
    sys.modules[PLUGIN_NAMESPACE] = namespace


def _config_defaults_from_source(path: Path) -> dict[str, Any]:
    """Lit ``CONFIG_DEFAULTS`` *avant* d'exécuter le fichier.

    Sans cela, un plugin qui appelle ``get_config()`` au niveau du module —
    ce que le contrat autorise explicitement — ne verrait rien : ses propres
    défauts sont définis par le fichier qu'on n'a pas encore exécuté. On les
    récupère donc par analyse syntaxique, sans rien évaluer.
    """
    try:
        arbre = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return {}
    for noeud in arbre.body:
        cibles = (
            noeud.targets if isinstance(noeud, ast.Assign)
            else [noeud.target] if isinstance(noeud, ast.AnnAssign) and noeud.value
            else []
        )
        if any(isinstance(c, ast.Name) and c.id == "CONFIG_DEFAULTS" for c in cibles):
            try:
                valeur = ast.literal_eval(noeud.value)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                return {}  # dict calculé : on se rabattra sur l'après-import
            return dict(valeur) if isinstance(valeur, dict) else {}
    return {}


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _missing_package_of(exc: BaseException) -> str | None:
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        return exc.name.split(".")[0]
    return None


def run_with_timeout(
    func: Callable[..., Any],
    kwargs: Mapping[str, Any],
    timeout: float | None,
) -> Any:
    """Exécute ``func`` dans un thread démon et abandonne au bout de ``timeout``.

    Le thread est démon pour qu'un plugin bloqué n'empêche jamais Capucine de
    s'arrêter. On attrape ``BaseException`` : un ``sys.exit()`` dans un plugin
    lève ``SystemExit``, qui n'hérite pas de ``Exception`` et tuerait le
    processus sans cette précaution.
    """
    outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            if inspect.iscoroutinefunction(func):
                import asyncio

                value = asyncio.run(func(**kwargs))
            else:
                value = func(**kwargs)
            outcome.put((True, value))
        except BaseException as exc:  # noqa: BLE001 - c'est précisément le but
            outcome.put((False, exc))

    worker = threading.Thread(target=target, name=f"skill-{getattr(func, '__name__', '?')}", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise SkillTimeout(f"délai de {timeout:g} s dépassé")
    try:
        succeeded, payload = outcome.get_nowait()
    except queue.Empty:  # pragma: no cover - ne devrait pas arriver
        raise SkillCrashed("le plugin n'a rien retourné") from None
    if succeeded:
        return payload
    if isinstance(payload, KeyboardInterrupt):
        raise payload
    raise SkillCrashed(str(payload) or type(payload).__name__) from payload


def normalize_result(value: Any, skill_name: str, duration_ms: float) -> SkillResult:
    """Applique la convention de retour du contrat de plugin.

    Une chaîne est lue telle quelle. Un dict ``{"speak": …, "display": …}``
    dissocie ce qui est dit de ce qui est journalisé.
    """
    if value is None:
        return SkillResult(ok=True, skill=skill_name, speak="", display="", duration_ms=duration_ms)
    if isinstance(value, str):
        return SkillResult(ok=True, skill=skill_name, speak=value, display=value, duration_ms=duration_ms)
    if isinstance(value, Mapping):
        speak = str(value.get("speak", "") or "")
        display = str(value.get("display", speak) or speak)
        return SkillResult(ok=True, skill=skill_name, speak=speak, display=display, duration_ms=duration_ms)
    text = str(value)
    return SkillResult(ok=True, skill=skill_name, speak=text, display=text, duration_ms=duration_ms)


class PluginRegistry:
    """Le registre : ce que Capucine sait faire, à un instant donné."""

    def __init__(
        self,
        paths: Iterable[Path],
        config: Any = None,
        *,
        default_timeout: float = DEFAULT_TIMEOUT,
        isolate_startup_s: float = 3.0,
        data_root: Path | None = None,
        quarantine_after: int = QUARANTINE_AFTER,
        on_change: Callable[[list[str], list[str]], None] | None = None,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.config = config
        self.default_timeout = default_timeout
        self.isolate_startup_s = isolate_startup_s
        self.data_root = Path(data_root) if data_root else Path.home() / ".capucine" / "data"
        self.quarantine_after = quarantine_after
        self.on_change = on_change

        self._lock = threading.RLock()
        self._skills: dict[str, SkillSpec] = {}
        self._plugins: dict[str, PluginRecord] = {}
        _ensure_namespace()

    # -- lecture ------------------------------------------------------------
    @property
    def skills(self) -> dict[str, SkillSpec]:
        """Copie de la table : le pipeline travaille sur un instantané, ce qui
        rend le rechargement à chaud invisible pour un tour en cours."""
        with self._lock:
            return dict(self._skills)

    @property
    def plugins(self) -> dict[str, PluginRecord]:
        with self._lock:
            return dict(self._plugins)

    def get(self, name: str) -> SkillSpec | None:
        with self._lock:
            return self._skills.get(name)

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [spec.tool_schema for spec in self.skills.values() if not spec.quarantined]

    def failures(self) -> list[PluginRecord]:
        return [record for record in self.plugins.values() if not record.ok]

    # -- découverte ---------------------------------------------------------
    def discover_files(self) -> list[Path]:
        """Les fichiers candidats, dans l'ordre des chemins configurés.

        Un fichier d'un chemin ultérieur masque un même nom d'un chemin
        antérieur : ``~/.capucine/plugins`` peut ainsi surcharger un plugin
        livré avec le projet.
        """
        found: dict[str, Path] = {}
        for directory in self.paths:
            if not directory.is_dir():
                logger.debug("Dossier de plugins absent : %s", directory)
                continue
            for path in sorted(directory.glob("*.py")):
                if path.name.startswith(("_", ".")):
                    continue
                stem = unicodedata.normalize("NFKC", path.stem)
                if stem in found:
                    logger.warning("Le plugin %s masque %s", path, found[stem])
                found[stem] = path
        return list(found.values())

    def load_all(self) -> None:
        """Charge tous les plugins découverts. N'échoue jamais globalement."""
        before = set(self.skills)
        for path in self.discover_files():
            self.load_file(path, notify=False)
        after = set(self.skills)
        self._notify(sorted(after - before), sorted(before - after))
        loaded = sum(1 for record in self.plugins.values() if record.ok)
        log_with(
            logger, logging.INFO, "Plugins chargés",
            plugins=loaded, echecs=len(self.failures()), competences=len(self._skills),
        )

    # -- chargement ---------------------------------------------------------
    def load_file(self, path: Path, *, notify: bool = True) -> PluginRecord:
        path = Path(path)
        module_name = _module_name_for(path)
        plugin_name = unicodedata.normalize("NFKC", path.stem)
        before = set(self.skills)

        # Les défauts sont lus par analyse syntaxique avant l'exécution, pour
        # que get_config() fonctionne dès le corps du module.
        defaults = _config_defaults_from_source(path)
        context = PluginContext(
            plugin=plugin_name,
            module=module_name,
            config=(
                self.config.plugin_config(plugin_name, defaults)
                if self.config is not None else dict(defaults)
            ),
            data_dir=self.data_root / plugin_name,
        )
        # Le contexte est publié AVANT l'exécution du fichier : un plugin peut
        # appeler get_config() au niveau du module, pas seulement dans un skill.
        register_context(context)

        record = PluginRecord(name=plugin_name, path=path, module_name=module_name, digest=_digest(path))

        try:
            module = self._import_module(path, module_name)
        except PluginImportError as exc:
            record.status = "echec"
            record.error = str(exc)
            record.missing_package = exc.missing_package
            clear_context(module_name)
            sys.modules.pop(module_name, None)
            with self._lock:
                self._plugins[plugin_name] = record
            logger.error("%s", exc)
            return record

        # Deuxième passe : si CONFIG_DEFAULTS est calculé plutôt que littéral,
        # l'analyse syntaxique l'a manqué. On refusionne avec la vraie valeur.
        defaults = dict(config_defaults(module)) or defaults
        context.config = (
            self.config.plugin_config(plugin_name, defaults) if self.config is not None else dict(defaults)
        )

        specs = self._collect_skills(module, path, plugin_name, module_name)

        hook_error = self._run_hook(module, "on_load", plugin_name)
        if hook_error is not None:
            record.status = "echec"
            record.error = hook_error
            clear_context(module_name)
            sys.modules.pop(module_name, None)
            with self._lock:
                self._plugins[plugin_name] = record
            return record

        record.skills = [spec.name for spec in specs]
        with self._lock:
            # Remplacement d'un bloc : on retire les anciens skills de ce
            # plugin puis on installe les nouveaux, sans état intermédiaire
            # visible depuis un tour en cours.
            table = {
                name: spec for name, spec in self._skills.items()
                if spec.plugin != plugin_name
            }
            for spec in specs:
                if spec.name in table:
                    logger.warning(
                        "La compétence « %s » de %s remplace celle de %s",
                        spec.name, plugin_name, table[spec.name].plugin,
                    )
                table[spec.name] = spec
            self._skills = table
            self._plugins[plugin_name] = record

        log_with(
            logger, logging.INFO, "Plugin chargé",
            plugin=plugin_name, competences=",".join(record.skills) or "-",
        )
        if notify:
            after = set(self.skills)
            self._notify(sorted(after - before), sorted(before - after))
        return record

    def _import_module(self, path: Path, module_name: str) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise PluginImportError(str(path), f"Fichier de plugin illisible : {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # nécessaire aux dataclasses et au typing
        try:
            # On compile la source nous-mêmes plutôt que d'appeler
            # `exec_module`. Raison : le cache de bytecode de CPython est
            # validé sur (date de modification, taille). Remplacer « Bonjour »
            # par « Bonsoir » dans la même seconde ne change ni l'une ni
            # l'autre — et le rechargement à chaud rejouerait silencieusement
            # l'ancien code. Passer par `compile` supprime le problème à la
            # racine, et évite d'écrire des .pyc dans le dossier des plugins.
            source = path.read_text(encoding="utf-8")
            code = compile(source, str(path), "exec")
            exec(code, module.__dict__)  # noqa: S102 - c'est précisément le but
        except BaseException as exc:  # noqa: BLE001 - un plugin ne fait jamais tomber Capucine
            if isinstance(exc, KeyboardInterrupt):
                raise
            sys.modules.pop(module_name, None)
            missing = _missing_package_of(exc)
            if missing:
                message = (
                    f"Plugin « {path.name} » ignoré : le paquet « {missing} » est absent. "
                    f"Installez-le avec : pip install {missing}"
                )
            else:
                message = f"Plugin « {path.name} » ignoré : {type(exc).__name__}: {exc}"
            logger.debug("Trace du plugin %s", path.name, exc_info=exc)
            raise PluginImportError(str(path), message, missing) from exc
        return module

    def _collect_skills(
        self, module: types.ModuleType, path: Path, plugin_name: str, module_name: str
    ) -> list[SkillSpec]:
        specs: list[SkillSpec] = []
        for attribute_name, value in vars(module).items():
            declaration = getattr(value, SKILL_ATTRIBUTE, None)
            if not isinstance(declaration, SkillDeclaration):
                continue
            if getattr(value, "__module__", module_name) != module_name:
                continue  # skill importé depuis un autre plugin : il appartient à l'autre
            try:
                spec = build_skill_spec(
                    value, declaration, module=module_name, source=path, plugin=plugin_name
                )
            except Exception as exc:  # schéma impossible : on écarte ce skill seul
                logger.error(
                    "Compétence « %s » de %s ignorée : %s", attribute_name, plugin_name, exc
                )
                continue
            if spec.timeout is None:
                spec.timeout = self.default_timeout
            specs.append(spec)
        return specs

    def _run_hook(self, module: types.ModuleType, hook: str, plugin_name: str) -> str | None:
        function = getattr(module, hook, None)
        if not callable(function):
            return None
        try:
            run_with_timeout(function, {}, self.default_timeout)
        except (SkillTimeout, SkillCrashed) as exc:
            message = f"Plugin « {plugin_name} » ignoré : {hook}() a échoué ({exc})"
            logger.error("%s", message)
            return message
        return None

    # -- déchargement -------------------------------------------------------
    def unload(self, plugin_name: str, *, notify: bool = True) -> bool:
        plugin_name = unicodedata.normalize("NFKC", plugin_name)
        with self._lock:
            record = self._plugins.get(plugin_name)
        if record is None:
            return False
        before = set(self.skills)

        module = sys.modules.get(record.module_name)
        if module is not None:
            self._run_hook(module, "on_unload", plugin_name)

        with self._lock:
            self._skills = {
                name: spec for name, spec in self._skills.items() if spec.plugin != plugin_name
            }
            self._plugins.pop(plugin_name, None)
        clear_context(record.module_name)
        sys.modules.pop(record.module_name, None)
        logger.info("Plugin déchargé : %s", plugin_name)
        if notify:
            after = set(self.skills)
            self._notify(sorted(after - before), sorted(before - after))
        return True

    def reload_file(self, path: Path) -> PluginRecord:
        """Décharge puis recharge un fichier. Point d'entrée de l'étape 4."""
        plugin_name = unicodedata.normalize("NFKC", Path(path).stem)
        before = set(self.skills)
        self.unload(plugin_name, notify=False)
        record = self.load_file(path, notify=False)
        after = set(self.skills)
        self._notify(sorted(after - before), sorted(before - after))
        return record

    def _notify(self, added: list[str], removed: list[str]) -> None:
        if self.on_change and (added or removed):
            try:
                self.on_change(added, removed)
            except Exception:  # pragma: no cover - une annonce ne casse rien
                logger.exception("Le rappel on_change a échoué.")

    # -- exécution ----------------------------------------------------------
    def call(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        confirmed: bool = False,
    ) -> SkillResult:
        """Exécute une compétence. Ne lève jamais.

        C'est la frontière : au-delà, un plugin peut faire n'importe quoi ; en
        deçà, le pipeline ne reçoit qu'un ``SkillResult``.

        Une compétence déclarée ``confirm=`` n'est pas exécutée du premier
        coup : elle rend un résultat qui **demande** confirmation, à charge du
        pipeline de poser la question et de rappeler avec ``confirmed=True``.
        """
        spec = self.get(name)
        if spec is None:
            return SkillResult.failure(name, f"Je ne connais pas la compétence « {name} ».")
        if spec.quarantined:
            return SkillResult.failure(
                name, f"La compétence « {name} » est désactivée après plusieurs échecs."
            )

        try:
            cleaned, dropped = coerce_arguments(spec.parameters_schema, dict(arguments or {}))
        except ArgumentError as exc:
            return SkillResult.failure(name, str(exc))
        if dropped:
            logger.debug("Arguments ignorés pour %s : %s", name, ", ".join(dropped))

        if spec.confirm and not confirmed:
            question = spec.confirmation_question()
            logger.info("Confirmation demandée avant « %s ».", name)
            return SkillResult(
                ok=True, skill=name, speak=question, display=question,
                needs_confirmation=True, arguments=cleaned,
            )

        started = time.perf_counter()
        try:
            value = self._invoquer(spec, cleaned)
        except SkillTimeout as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            self._record_failure(spec)
            logger.error("Compétence « %s » abandonnée : %s", name, exc)
            return SkillResult.failure(
                name, "Je n'ai pas pu exécuter cette commande, elle a mis trop de temps.", elapsed
            )
        except SkillCrashed as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            self._record_failure(spec)
            logger.error("Compétence « %s » en échec : %s", name, exc, exc_info=exc.__cause__)
            return SkillResult.failure(name, "Je n'ai pas pu exécuter cette commande.", elapsed)

        elapsed = (time.perf_counter() - started) * 1000.0
        spec.failures = 0
        return normalize_result(value, name, round(elapsed, 1))

    def _invoquer(self, spec: SkillSpec, arguments: dict[str, Any]) -> Any:
        """Exécute la compétence, en thread ou en sous-processus.

        Le thread est le défaut : il est instantané et laisse le plugin garder
        son état d'un appel à l'autre. ``isolate=True`` échange cet état et
        100 à 300 ms de démarrage contre la seule garantie qui compte pour un
        traitement capable de bloquer indéfiniment : un processus, lui, se tue.
        """
        if not spec.isolate:
            return run_with_timeout(spec.func, arguments, spec.timeout)

        from .isolation import run_isolated
        from .plugin import contexte_de

        contexte = contexte_de(spec.module)
        return run_isolated(
            spec.source,
            spec.module,
            getattr(spec.func, "__name__", spec.name),
            arguments,
            timeout=spec.timeout,
            startup_s=self.isolate_startup_s,
            plugin=spec.plugin,
            config=dict(contexte.config) if contexte else {},
            data_dir=contexte.data_dir if contexte else None,
        )

    def _record_failure(self, spec: SkillSpec) -> None:
        spec.failures += 1
        if self.quarantine_after and spec.failures >= self.quarantine_after:
            spec.quarantined = True
            logger.error(
                "Compétence « %s » mise en quarantaine après %d échecs.", spec.name, spec.failures
            )

    def reset_quarantine(self, name: str) -> None:
        spec = self.get(name)
        if spec is not None:
            spec.failures = 0
            spec.quarantined = False

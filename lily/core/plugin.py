"""Le contrat public offert aux plugins.

Un plugin est un fichier ``.py`` déposé dans ``plugins/``. Il n'a besoin que
de ce module :

    from lily.plugin import skill

    @skill(description="…", examples=["…"])
    def ma_competence(argument: str) -> str:
        \"\"\"Cette docstring sert de contexte au LLM.\"\"\"
        return "réponse lue à voix haute"

Rien d'autre : pas d'enregistrement manuel, pas d'import à ajouter ailleurs,
pas de manifeste. Le module ``lily.plugin`` n'est qu'une façade sur ce
fichier, pour que le cœur puisse bouger sans casser un seul plugin.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import SchemaError, SkillRefused
from .logging import get_logger
from .schema import build_parameters_schema, build_tool_schema
from .text import ascii_identifier

__all__ = [
    "skill",
    "SkillRefused",
    "SkillDeclaration",
    "SkillSpec",
    "PluginContext",
    "get_config",
    "get_logger_for_plugin",
    "data_dir",
    "announce",
    "set_announcer",
    "demander_au_modele",
    "set_model_access",
    "atelier",
    "set_atelier",
    "memoire",
    "set_memoire",
    "conversation",
    "set_conversation",
    "apprentissage",
    "set_apprentissage",
    "connaissances",
    "set_connaissances",
    "corpus",
    "set_corpus",
    "journal",
    "set_journal",
    "appeler_competence",
    "set_registre",
    "dossier_des_plugins",
    "set_dossier_des_plugins",
    "pile_d_appels",
    "adopter_la_pile",
    "catalogue",
    "set_catalogue",
    "register_context",
    "clear_context",
    "contexte_de",
    "SKILL_ATTRIBUTE",
]

SKILL_ATTRIBUTE = "__lily_skill__"


@dataclass(frozen=True)
class SkillDeclaration:
    """Ce que le décorateur attache à la fonction, avant tout chargement."""

    description: str
    examples: tuple[str, ...]
    name: str | None
    timeout: float | None
    confirm: bool | str
    isolate: bool = False


def skill(
    _func: Callable[..., Any] | None = None,
    *,
    description: str = "",
    examples: list[str] | tuple[str, ...] | None = None,
    name: str | None = None,
    timeout: float | None = None,
    confirm: bool | str = False,
    isolate: bool = False,
) -> Callable[..., Any]:
    """Déclare une fonction comme compétence de Lily.

    Args:
        description: Une phrase, en français, qui dit ce que fait la
            compétence. Elle part telle quelle dans le contexte du LLM.
        examples: Formulations typiques de l'utilisateur. Elles ne sont pas
            décoratives : le routeur déterministe s'en sert pour choisir
            l'outil sans solliciter le modèle.
        name: Nom d'outil, si le nom de la fonction ne convient pas.
        timeout: Délai maximum d'exécution, en secondes. Au-delà, Lily
            répond qu'elle n'a pas pu exécuter la commande.
        confirm: Pour les actions irréversibles. ``True`` fait poser une
            question générique avant d'exécuter ; une chaîne fournit la
            question exacte (« Voulez-vous vraiment effacer toutes les
            notes ? »).
        isolate: Exécute la compétence dans un sous-processus, réellement
            tuable au bout du délai. Coûte 100 à 300 ms par appel, interdit
            l'état en mémoire, et exige des arguments et un retour
            sérialisables. À réserver aux traitements capables de bloquer
            indéfiniment.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(func):  # pragma: no cover - garde-fou
            raise SchemaError("@skill s'applique à une fonction.")
        setattr(
            func,
            SKILL_ATTRIBUTE,
            SkillDeclaration(
                description=description.strip(),
                examples=tuple(examples or ()),
                name=name,
                timeout=timeout,
                confirm=confirm,
                isolate=isolate,
            ),
        )
        return func

    if _func is not None:
        return decorator(_func)
    return decorator


@dataclass
class SkillSpec:
    """Une compétence chargée et prête à être appelée."""

    name: str
    func: Callable[..., Any]
    module: str
    source: Path
    description: str
    examples: tuple[str, ...]
    timeout: float | None
    confirm: bool | str
    isolate: bool
    tool_schema: dict[str, Any]
    parameters_schema: dict[str, Any]
    plugin: str
    qualname: str = ""
    failures: int = 0
    quarantined: bool = False

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.func)

    @property
    def required_parameters(self) -> list[str]:
        return list(self.parameters_schema.get("required", []))

    @property
    def parameter_names(self) -> list[str]:
        return list(self.parameters_schema.get("properties", {}))

    def confirmation_question(self) -> str:
        """La question posée avant d'exécuter une action irréversible."""
        if isinstance(self.confirm, str) and self.confirm.strip():
            return self.confirm.strip()
        return f"Voulez-vous vraiment que j'exécute « {self.name.replace('_', ' ')} » ?"

    def matchable_phrases(self) -> list[str]:
        """Ce que le routeur déterministe compare à la phrase entendue."""
        readable_name = self.name.replace("_", " ")
        return [readable_name, *self.examples, self.description]


def build_skill_spec(
    func: Callable[..., Any],
    declaration: SkillDeclaration,
    *,
    module: str,
    source: Path,
    plugin: str,
) -> SkillSpec:
    tool_name = ascii_identifier(declaration.name or func.__name__)
    parameters_schema, _ = build_parameters_schema(func)
    tool_schema = build_tool_schema(
        func,
        name=tool_name,
        description=declaration.description,
        examples=list(declaration.examples),
    )
    return SkillSpec(
        name=tool_name,
        func=func,
        module=module,
        source=source,
        description=tool_schema["function"]["description"],
        examples=declaration.examples,
        timeout=declaration.timeout,
        confirm=declaration.confirm,
        isolate=declaration.isolate,
        tool_schema=tool_schema,
        parameters_schema=parameters_schema,
        plugin=plugin,
        qualname=getattr(func, "__qualname__", func.__name__),
    )


# --- contexte d'un plugin --------------------------------------------------

@dataclass
class PluginContext:
    """Ce que le cœur met à disposition d'un plugin donné."""

    plugin: str
    module: str
    config: dict[str, Any] = field(default_factory=dict)
    data_dir: Path | None = None
    logger: logging.Logger | None = None

    def get_logger(self) -> logging.Logger:
        if self.logger is None:
            self.logger = get_logger(f"plugin.{self.plugin}")
        return self.logger


_CONTEXTS: dict[str, PluginContext] = {}
_ANNOUNCER: Callable[[str], None] | None = None
_logger = get_logger("plugin")


def register_context(context: PluginContext) -> None:
    _CONTEXTS[context.module] = context


def clear_context(module: str) -> None:
    _CONTEXTS.pop(module, None)


def contexte_de(module: str) -> PluginContext | None:
    """Contexte enregistré pour un module donné, s'il existe."""
    return _CONTEXTS.get(module)


def _caller_context(depth: int = 2) -> PluginContext | None:
    """Retrouve le plugin appelant en remontant la pile.

    Fonctionne à l'import du module comme à l'exécution d'un skill, parce que
    le registre enregistre le contexte *avant* d'exécuter le fichier.
    """
    frame = inspect.currentframe()
    for _ in range(depth):
        if frame is None:
            return None
        frame = frame.f_back
    while frame is not None:
        module_name = frame.f_globals.get("__name__", "")
        context = _CONTEXTS.get(module_name)
        if context is not None:
            return context
        frame = frame.f_back
    return None


def get_config(key: str | None = None, default: Any = None) -> Any:
    """Configuration du plugin appelant.

    Sans argument, retourne le dictionnaire complet : ``CONFIG_DEFAULTS``
    du module surchargé par la section ``[plugins.<nom>]`` du fichier de
    configuration.
    """
    context = _caller_context()
    if context is None:
        _logger.debug("get_config() hors contexte de plugin.")
        return default if key is not None else {}
    if key is None:
        return dict(context.config)
    return context.config.get(key, default)


def get_logger_for_plugin() -> logging.Logger:
    """Journal nommé d'après le plugin appelant."""
    context = _caller_context()
    return context.get_logger() if context else get_logger("plugin.inconnu")


def data_dir() -> Path:
    """Dossier inscriptible réservé au plugin appelant, créé à la demande."""
    context = _caller_context()
    if context is None or context.data_dir is None:
        fallback = Path.home() / ".lily" / "data" / "inconnu"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    context.data_dir.mkdir(parents=True, exist_ok=True)
    return context.data_dir


def set_announcer(announcer: Callable[[str], None] | None) -> None:
    """Branche la sortie vocale utilisée par ``announce()``.

    Le pipeline l'installe au démarrage ; en dehors, ``announce()`` se replie
    sur le journal, ce qui rend les plugins testables sans assistant.
    """
    global _ANNOUNCER
    _ANNOUNCER = announcer


def announce(message: str) -> None:
    """Fait dire quelque chose à Lily hors d'un tour de conversation.

    C'est ce dont a besoin une tâche de fond : un minuteur qui arrive à
    échéance n'a personne à qui répondre, il doit *interrompre*.
    """
    if _ANNOUNCER is not None:
        try:
            _ANNOUNCER(message)
            return
        except Exception:  # pragma: no cover - une annonce ne casse jamais rien
            _logger.exception("Échec de l'annonce, repli sur le journal.")
    _logger.info("[annonce] %s", message)


# --- accès au modèle, à l'atelier et à la mémoire --------------------------
# Trois ressources que le cœur détient et qu'un plugin peut demander. Elles
# passent par des points d'injection, comme `announce` : un plugin ne connaît
# jamais le pipeline, et reste testable en le remplaçant.

_MODELE: Callable[..., str] | None = None
_ATELIER: Any = None
_MEMOIRE: Any = None
_CONVERSATION: Any = None
_APPRENTISSAGE: Any = None
_CONNAISSANCES: Any = None
_CORPUS: Any = None
_REGISTRE: Any = None
_JOURNAL: Any = None
_DOSSIER_PLUGINS: Any = None
_CATALOGUE: Any = None
# Profondeur d'imbrication autorisée quand une compétence en appelle une
# autre. Trois suffit à composer des routines ; au-delà, c'est une boucle.
PROFONDEUR_MAX = 3
_pile = threading.local()


def set_model_access(fonction: Callable[..., str] | None) -> None:
    """Branche l'accès au modèle de langage utilisé par ``demander_au_modele``."""
    global _MODELE
    _MODELE = fonction


def demander_au_modele(
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = 512,
    temperature: float = 0.2,
    json_schema: dict[str, Any] | None = None,
) -> str:
    """Pose une question au modèle de langage local, et rend sa réponse brute.

    C'est ce qui permet à un plugin d'**écrire du code** ou de résumer un
    texte. Volontairement limité à une complétion simple : pas de routage,
    pas d'outils, donc pas de récursion possible — un plugin ne peut pas
    déclencher un autre plugin par ce biais.

    Lève ``RuntimeError`` si aucun modèle n'est joignable ; le plugin doit le
    dire à l'utilisateur plutôt que d'inventer une réponse.
    """
    if _MODELE is None:
        raise SkillRefused(
            "Aucun modèle de langage n'est disponible. Vérifiez llm.engine dans "
            "la configuration, et qu'Ollama répond."
        )
    return _MODELE(
        prompt, system=system, max_tokens=max_tokens,
        temperature=temperature, json_schema=json_schema,
    )


def set_atelier(instance: Any) -> None:
    """Branche l'atelier — les dossiers que Lily a le droit de toucher."""
    global _ATELIER
    _ATELIER = instance


def atelier() -> Any:
    """L'atelier courant. Lève si aucun dossier n'a été ouvert.

    Tous les accès disque d'un plugin devraient passer par là : c'est le seul
    endroit qui vérifie qu'un chemin reste dans les dossiers autorisés.
    """
    if _ATELIER is None or not getattr(_ATELIER, "ouvert", False):
        raise SkillRefused(
            "Aucun dossier de travail n'est ouvert. Renseignez atelier.racines "
            "dans la configuration — par sécurité, la liste est vide au départ."
        )
    return _ATELIER


def set_memoire(instance: Any) -> None:
    """Branche la mémoire persistante."""
    global _MEMOIRE
    _MEMOIRE = instance


def set_apprentissage(instance: Any) -> None:
    """Branche le magasin de ce qui s'apprend au fil des tours."""
    global _APPRENTISSAGE
    _APPRENTISSAGE = instance


def apprentissage() -> Any:
    """Ce que Lily a retenu de vos formulations et de votre vocabulaire."""
    if _APPRENTISSAGE is None:
        raise SkillRefused(
            "L'apprentissage est désactivé (apprentissage.active = false)."
        )
    return _APPRENTISSAGE


def set_connaissances(instance: Any) -> None:
    """Branche l'index sémantique des documents et des conversations."""
    global _CONNAISSANCES
    _CONNAISSANCES = instance


def connaissances() -> Any:
    """L'index de ce que Lily a lu : documents indexés, tours passés.

    Un plugin y dépose du texte (``indexer``) et y pose des questions
    (``chercher``, ``contexte``). Sans vectoriseur, la recherche se rabat sur
    le plein texte : le plugin n'a pas à s'en soucier.
    """
    if _CONNAISSANCES is None:
        raise SkillRefused(
            "L'index des connaissances est désactivé (connaissances.active = false)."
        )
    return _CONNAISSANCES


def set_corpus(instance: Any) -> None:
    """Branche le corpus d'éveil — les extraits gardés autour des détections."""
    global _CORPUS
    _CORPUS = instance


def corpus() -> Any:
    """Le corpus d'éveil, allumé ou non.

    Contrairement aux autres ressources, il est rendu même éteint : une
    compétence doit pouvoir expliquer comment l'allumer plutôt que de refuser
    sans rien dire.
    """
    if _CORPUS is None:
        raise SkillRefused(
            "Le corpus d'éveil n'existe pas dans cette configuration "
            "(section [corpus] absente)."
        )
    return _CORPUS


def set_journal(instance: Any) -> None:
    """Branche le journal des compétences appelées."""
    global _JOURNAL
    _JOURNAL = instance


def journal() -> Any:
    """Les dernières compétences exécutées — de quoi apprendre une routine."""
    if _JOURNAL is None:
        raise SkillRefused("Le journal des appels n'est pas disponible.")
    return _JOURNAL


def set_registre(instance: Any) -> None:
    """Branche le registre, pour qu'une compétence puisse en appeler une autre."""
    global _REGISTRE
    _REGISTRE = instance


def pile_d_appels() -> list[str]:
    """Les compétences actuellement empilées sur ce fil d'exécution."""
    return list(getattr(_pile, "appels", None) or [])


def adopter_la_pile(pile: list[str]) -> None:
    """Reprend la pile d'un autre fil.

    Le registre exécute chaque compétence dans un thread neuf : sans ce
    passage de témoin, la pile serait vide à l'intérieur et le garde-fou de
    récursion ne verrait jamais rien.
    """
    _pile.appels = list(pile)


def appeler_competence(nom: str, arguments: dict[str, Any] | None = None) -> Any:
    """Exécute une autre compétence et rend son ``SkillResult``.

    C'est ce qui rend les routines possibles : un plugin qui enchaîne
    « l'heure, puis mes notes, puis l'état du système » sans réimplémenter
    aucun des trois.

    Deux garde-fous, parce qu'ouvrir cette porte est délicat :

    * **La profondeur est bornée.** Une compétence qui s'appelle elle-même,
      directement ou par une chaîne de routines, est arrêtée avant de remplir
      la pile — et le refus est dit à voix haute plutôt que de faire tomber
      Lily.
    * **La confirmation n'est jamais contournée.** Une compétence déclarée
      ``confirm=`` rend sa demande telle quelle : une routine ne peut pas
      effacer vos notes en passant.
    """
    if _REGISTRE is None:
        raise SkillRefused("Aucun registre de compétences n'est disponible.")
    pile: list[str] = getattr(_pile, "appels", None) or []
    if len(pile) >= PROFONDEUR_MAX:
        raise SkillRefused(
            f"Trop d'appels imbriqués ({' → '.join([*pile, nom])}). "
            "Une routine ne peut pas en enchaîner indéfiniment d'autres."
        )
    if nom in pile:
        raise SkillRefused(
            f"« {nom} » s'appelle elle-même ({' → '.join([*pile, nom])})."
        )
    _pile.appels = [*pile, nom]
    try:
        return _REGISTRE.call(nom, arguments or {})
    finally:
        _pile.appels = pile


def set_dossier_des_plugins(chemin: Any) -> None:
    """Branche le dossier où déposer un plugin écrit par Lily elle-même."""
    global _DOSSIER_PLUGINS
    _DOSSIER_PLUGINS = Path(chemin) if chemin is not None else None


def dossier_des_plugins() -> Path:
    """Le dossier ``plugins/`` surveillé, pour qu'elle puisse s'y écrire.

    C'est la contrainte numéro un du projet retournée comme un gant : ajouter
    une capacité, c'est déposer un fichier ici — y compris quand c'est
    Lily qui le dépose.
    """
    if _DOSSIER_PLUGINS is None or not _DOSSIER_PLUGINS.is_dir():
        raise SkillRefused(
            "Je ne sais pas où écrire un plugin : aucun dossier de plugins "
            "accessible en écriture."
        )
    return _DOSSIER_PLUGINS


def set_catalogue(instance: Any) -> None:
    """Branche le catalogue d'API — les signatures de VOS fonctions."""
    global _CATALOGUE
    _CATALOGUE = instance


def catalogue() -> Any:
    """Les signatures du dépôt courant, pour que le modèle n'invente pas d'API.

    C'est le contrat de ``schema.py`` — signature plus docstring égale contrat
    — appliqué au code que vous écrivez, et plus seulement aux compétences.
    """
    if _CATALOGUE is None:
        raise SkillRefused(
            "Aucun catalogue d'API n'est disponible. Renseignez catalogue.racine "
            "dans la configuration, ou lancez avec --atelier sur un dépôt."
        )
    return _CATALOGUE


def set_conversation(instance: Any) -> None:
    """Branche le fil de conversation courant."""
    global _CONVERSATION
    _CONVERSATION = instance


def conversation() -> Any:
    """Le fil courant. Permet à un plugin de reprendre une session passée.

    C'est la seule ressource qui donne prise sur l'état du dialogue : elle
    reste volontairement étroite, un plugin ne peut que relire ou remplacer
    le fil, jamais s'insérer dans le routage.
    """
    if _CONVERSATION is None:
        raise RuntimeError("Aucune conversation en cours.")
    return _CONVERSATION


def memoire() -> Any:
    """La mémoire persistante. Lève si elle est désactivée."""
    if _MEMOIRE is None:
        raise SkillRefused(
            "La mémoire persistante est désactivée (memoire.active = false)."
        )
    return _MEMOIRE


def config_defaults(module: Any) -> Mapping[str, Any]:
    values = getattr(module, "CONFIG_DEFAULTS", {})
    return values if isinstance(values, Mapping) else {}

"""Assemblage : configuration → registre → routeur → pipeline.

C'est le seul endroit qui connaît tous les étages à la fois. Tout le reste du
cœur n'en voit qu'un.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass
from typing import Any

from .core.atelier import Atelier
from .core.atelier import depuis_config as atelier_depuis_config
from .core.audio import AudioInput, AudioOutput
from .core.config import Config
from .core.conversation import Conversation, load_persona
from .core.endpointer import BargeInDetector, Endpointer
from .core.engines.factory import (
    build_audio_input,
    build_audio_output,
    build_llm,
    build_stt,
    build_tts,
    build_vad,
    build_wake,
)
from .core.interfaces.llm import LLMEngine, Message
from .core.interfaces.stt import STTEngine
from .core.interfaces.tts import TTSEngine
from .core.interfaces.vad import VADEngine
from .core.interfaces.wake import WakeWordEngine
from .core.listener import BargeInMode, ListenMode, VoiceListener
from .core.logging import get_latency_book, get_logger
from .core.machine import conseils, decrire
from .core.memoire import Memoire
from .core.memoire import depuis_config as memoire_depuis_config
from .core.pipeline import Pipeline
from .core.plugin import set_atelier, set_conversation, set_memoire, set_model_access
from .core.registry import PluginRegistry
from .core.router import Router
from .core.watcher import PluginWatcher

logger = get_logger("app")


@dataclass
class Assistant:
    config: Config
    llm: LLMEngine
    registry: PluginRegistry
    router: Router
    conversation: Conversation
    pipeline: Pipeline
    stt: STTEngine | None = None
    tts: TTSEngine | None = None
    audio_in: AudioInput | None = None
    audio_out: AudioOutput | None = None
    vad: VADEngine | None = None
    wake: WakeWordEngine | None = None
    listener: VoiceListener | None = None
    watcher: PluginWatcher | None = None
    memoire: Memoire | None = None
    atelier: Atelier | None = None

    async def aclose(self) -> None:
        if self.memoire is not None:
            if self.conversation.session_id is not None:
                self.memoire.fermer_session(self.conversation.session_id)
            self.memoire.fermer()
        # On débranche les ressources prêtées aux plugins : un test qui monte
        # deux assistants ne doit pas hériter de l'atelier du précédent.
        set_model_access(None)
        set_atelier(None)
        set_memoire(None)
        set_conversation(None)
        if self.watcher is not None:
            self.watcher.stop()
        if self.listener is not None:
            self.listener.stop()
        await self.pipeline.aclose()
        for moteur in (self.vad, self.wake):
            if moteur is not None:
                moteur.close()
        self.llm.close()


def build_assistant(
    config: Config,
    llm: LLMEngine | None = None,
    *,
    voice: bool = False,
    reprendre: str | None = None,
    stt: STTEngine | None = None,
    tts: TTSEngine | None = None,
    audio_in: AudioInput | None = None,
    audio_out: AudioOutput | None = None,
    wav_in: str | None = None,
    wav_out: str | None = None,
    silent_output: bool = False,
) -> Assistant:
    """Monte l'assistant. ``voice=True`` ajoute les étages audio.

    Les moteurs passés explicitement l'emportent : c'est ce qui permet aux
    tests de dérouler tout le chemin vocal avec des doublures.
    """
    engine = llm if llm is not None else build_llm(config)

    router_options = config.section("llm").get("router", {}) or {}
    router = Router(
        engine,
        direct_threshold=float(router_options.get("direct_threshold", 0.72)),
        shortlist_threshold=float(router_options.get("shortlist_threshold", 0.35)),
        shortlist_size=int(router_options.get("shortlist_size", 5)),
        allow_number_extraction=bool(router_options.get("number_extraction", True)),
    )

    memoire = memoire_depuis_config(config)
    conversation = Conversation(
        persona=load_persona(config.resolve_path("assistant.persona_file")),
        max_turns=int(config.get("assistant.memory_turns", 6)),
        memoire=memoire,
    )
    if memoire is not None:
        demande = reprendre
        if demande is None and bool(config.get("memoire.reprendre_au_demarrage", False)):
            demande = "derniere"
        if demande is not None:
            _reprendre_conversation(conversation, memoire, demande)
        else:
            conversation.session_id = memoire.ouvrir_session().id

    # Les trois ressources que le cœur prête aux plugins. `demander_au_modele`
    # est une complétion simple : jamais de routage, donc pas de récursion.
    atelier = atelier_depuis_config(config)
    set_atelier(atelier)
    set_memoire(memoire)
    set_conversation(conversation)
    set_model_access(_acces_modele(engine))

    registry = PluginRegistry(
        config.plugin_paths(),
        config=config,
        default_timeout=float(config.get("plugins.timeout", 10.0)),
        isolate_startup_s=float(config.get("plugins.isolate_startup_s", 3.0)),
        data_root=config.resolve_path("plugins.data_dir"),
        quarantine_after=int(config.get("plugins.quarantine_after", 3)),
    )

    if voice:
        stt = stt if stt is not None else build_stt(config)
        tts = tts if tts is not None else build_tts(config)
        audio_in = audio_in if audio_in is not None else build_audio_input(config, wav=wav_in)
        if audio_out is None:
            audio_out = build_audio_output(config, wav=wav_out, silent=silent_output)

    pipeline = Pipeline(
        registry,
        router,
        conversation,
        stt=stt,
        tts=tts,
        audio_in=audio_in,
        audio_out=audio_out,
        announce_new_skills=bool(config.get("assistant.announce_new_skills", True)),
        max_utterance_s=float(config.get("vad.max_utterance_s", 20.0)),
        follow_up_s=float(config.get("assistant.follow_up_seconds", 8.0)),
        wake_beep=bool(config.get("audio.wake_beep", True)),
    )

    registry.load_all()
    # Le rappel n'est branché qu'APRÈS le chargement initial : annoncer à voix
    # haute les vingt compétences déjà présentes au démarrage n'a aucun sens.
    # Le registre prévient le pipeline, qui décide quoi annoncer — le registre
    # n'a pas à savoir que Capucine a une voix.
    registry.on_change = pipeline.notify_skill_change

    return Assistant(
        config=config, llm=engine, registry=registry, router=router,
        conversation=conversation, pipeline=pipeline,
        stt=stt, tts=tts, audio_in=audio_in, audio_out=audio_out,
        memoire=memoire, atelier=atelier,
    )


def _acces_modele(engine: LLMEngine):
    """Adapte le moteur LLM à la signature simple offerte aux plugins."""

    def demander(prompt, *, system="", max_tokens=512, temperature=0.2, json_schema=None):
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        return engine.chat(
            messages, json_schema=json_schema,
            temperature=temperature, max_tokens=max_tokens,
        )

    return demander


def _reprendre_conversation(conversation: Conversation, memoire: Memoire, quoi: str) -> None:
    """``--reprendre derniere`` ou ``--reprendre 12``."""
    cible = None
    if quoi in ("derniere", "dernière", "last"):
        cible = memoire.derniere_session()
    elif quoi.isdigit():
        cible = memoire.session(int(quoi))
    if cible is None:
        logger.warning("Aucune conversation à reprendre pour « %s ».", quoi)
        conversation.session_id = memoire.ouvrir_session().id
        return
    conversation.reprendre(cible.id)


def start_hot_reload(assistant: Assistant) -> bool:
    """Démarre la surveillance de ``plugins/``. À appeler une fois la boucle
    asyncio en place, pour que les annonces vocales aient où aller."""
    config = assistant.config
    if not bool(config.get("plugins.hot_reload", True)):
        logger.info("Rechargement à chaud désactivé par la configuration.")
        return False
    assistant.watcher = PluginWatcher(
        assistant.registry,
        debounce_ms=float(config.get("plugins.debounce_ms", 500)),
    )
    return assistant.watcher.start()


def build_listener(
    assistant: Assistant,
    *,
    vad: VADEngine | None = None,
    barge_in_vad: VADEngine | None = None,
    wake: WakeWordEngine | None = None,
    use_wake: bool = True,
) -> VoiceListener:
    """Assemble le fil qui tient le micro : VAD, découpeur, éveil, barge-in."""
    config = assistant.config
    assistant.vad = vad if vad is not None else build_vad(config)
    if use_wake:
        assistant.wake = wake if wake is not None else build_wake(config)
    else:
        assistant.wake = wake

    endpointer = Endpointer(
        assistant.vad,
        threshold=float(config.get("vad.threshold", 0.5)),
        min_speech_ms=float(config.get("vad.min_speech_ms", 200)),
        silence_ms=float(config.get("vad.silence_ms", 700)),
        pre_roll_ms=float(config.get("vad.pre_roll_ms", 300)),
        max_utterance_s=float(config.get("vad.max_utterance_s", 20.0)),
        max_wait_s=float(config.get("vad.max_wait_s", 8.0)),
        min_total_speech_ms=float(config.get("vad.min_total_speech_ms", 300)),
    )
    # Le barge-in a son propre détecteur : même modèle, seuils différents.
    # Surtout, sa **propre instance** de VAD — Silero porte un état récurrent
    # d'une trame à l'autre, et le partager mélangerait l'écoute de
    # l'utilisateur avec la surveillance pendant la réponse.
    barge_in = BargeInDetector(
        barge_in_vad if barge_in_vad is not None else build_vad(config),
        threshold=float(config.get("barge_in.threshold", 0.85)),
        min_speech_ms=float(config.get("barge_in.min_speech_ms", 300)),
        guard_ms=float(config.get("barge_in.guard_ms", 400)),
    )

    if assistant.audio_in is None:
        raise RuntimeError("L'écoute continue réclame une entrée audio.")

    assistant.listener = VoiceListener(
        assistant.audio_in,
        endpointer=endpointer,
        on_event=assistant.pipeline.on_listener_event,
        wake=assistant.wake,
        barge_in=barge_in,
        barge_in_mode=BargeInMode(str(config.get("barge_in.mode", "voix"))),
        start_mode=ListenMode.PAUSED,
    )
    return assistant.listener


# --- commandes communes aux deux modes -------------------------------------

AIDE = """Commandes : /aide  /competences  /plugins  /recharge  /latences  /machine
            /conversations  /reprendre [n]  /memoire  /atelier  /oublie  /quitter
En mode vocal, une ligne vide déclenche l'écoute ; tout autre texte est traité
comme si vous l'aviez dit."""


def _format_skills(assistant: Assistant) -> str:
    skills = assistant.registry.skills
    if not skills:
        return "Aucune compétence chargée."
    lines = []
    for name, spec in sorted(skills.items()):
        signature = ", ".join(spec.parameter_names) or "sans argument"
        marque = " [en quarantaine]" if spec.quarantined else ""
        lines.append(f"  {name}({signature}) — {spec.plugin}{marque}")
    return "Compétences :\n" + "\n".join(lines)


def _format_plugins(assistant: Assistant) -> str:
    records = assistant.registry.plugins
    if not records:
        return "Aucun plugin trouvé. Vérifiez plugins.paths dans la configuration."
    lines = []
    for name, record in sorted(records.items()):
        if record.ok:
            lines.append(f"  ✓ {name} — {len(record.skills)} compétence(s) — {record.path}")
        else:
            lines.append(f"  ✗ {name} — {record.error}")
    return "Plugins :\n" + "\n".join(lines)


def _handle_command(assistant: Assistant, line: str) -> bool:
    """Retourne True s'il faut quitter."""
    command = line.split()[0].lower()
    if command in ("/quitter", "/quit", "/q"):
        return True
    if command in ("/aide", "/help", "/h"):
        print(AIDE)
    elif command in ("/competences", "/skills"):
        print(_format_skills(assistant))
    elif command == "/plugins":
        print(_format_plugins(assistant))
    elif command in ("/recharge", "/reload"):
        assistant.registry.load_all()
        print(_format_skills(assistant))
    elif command in ("/latences", "/latence"):
        print(get_latency_book().table())
        print("\nPour un relevé complet : python tools/mesurer_latence.py")
    elif command in ("/machine", "/materiel"):
        machine = decrire()
        print(f"Machine : {machine.resume()}")
        print(f"Profil actif : {assistant.config.get('profile')} "
              f"(conseillé : {machine.profil_conseille})")
        remarques = conseils(assistant.config, machine)
        print("\n".join(f"  • {remarque}" for remarque in remarques)
              or "  Rien à signaler.")
    elif command in ("/conversations", "/sessions"):
        if assistant.memoire is None:
            print("Mémoire persistante désactivée.")
        else:
            sessions = assistant.memoire.sessions(limite=10)
            print("\n".join(s.decrire() for s in sessions) or "Aucune conversation.")
    elif command in ("/reprendre", "/reprend"):
        if assistant.memoire is None:
            print("Mémoire persistante désactivée.")
        else:
            morceaux = line.split()
            _reprendre_conversation(
                assistant.conversation, assistant.memoire,
                morceaux[1] if len(morceaux) > 1 else "derniere",
            )
            print(f"Fil courant : {len(assistant.conversation)} message(s).")
    elif command in ("/memoire", "/faits"):
        if assistant.memoire is None:
            print("Mémoire persistante désactivée.")
        else:
            print(assistant.memoire.bloc_de_faits() or "Aucun fait retenu.")
    elif command == "/atelier":
        if assistant.atelier is None or not assistant.atelier.ouvert:
            print("Aucun dossier ouvert. Renseignez atelier.racines, "
                  "ou lancez avec --atelier CHEMIN.")
        else:
            print(f"Atelier : {assistant.atelier.decrire()}")
    elif command in ("/oublie", "/clear"):
        assistant.conversation.clear()
        print("Mémoire de conversation vidée.")
    else:
        print(f"Commande inconnue : {command}\n{AIDE}")
    return False


def _annoncer_demarrage(assistant: Assistant) -> None:
    print(_format_skills(assistant))
    for record in assistant.registry.failures():
        print(f"  ! plugin ignoré : {record.name} — {record.error}")
    # Mieux vaut prévenir que laisser découvrir qu'une transcription prend
    # douze secondes : les remarques matérielles sortent au démarrage.
    for remarque in conseils(assistant.config):
        print(f"  ⚠ {remarque}")
    for remarque in _remarques_atelier(assistant):
        print(f"  ⚠ {remarque}")


def _remarques_atelier(assistant: Assistant) -> list[str]:
    """Dit tout de suite si l'atelier est fermé, et pourquoi.

    Sans cela, les treize compétences qui touchent au disque refusent une par
    une sans qu'on comprenne d'où vient le problème — alors que la cause tient
    en une ligne : le dossier demandé n'existe pas.
    """
    espace = assistant.atelier
    if espace is None or espace.ouvert:
        return []

    concernees = sorted({
        spec.plugin for spec in assistant.registry.skills.values()
        if spec.plugin in ("fichiers", "python", "projet")
    })
    suffixe = (
        f" Les compétences {', '.join(concernees)} refuseront." if concernees else ""
    )
    if espace.racines_ignorees:
        return [
            "Dossier de travail introuvable : "
            + ", ".join(espace.racines_ignorees)
            + f". Vérifiez le chemin.{suffixe}"
        ]
    return [
        "Aucun dossier de travail ouvert (atelier.racines est vide). "
        f"Lancez avec --atelier CHEMIN, ou renseignez la configuration.{suffixe}"
    ]


# --- mode texte ------------------------------------------------------------

async def run_text_mode(assistant: Assistant, once: str | None = None) -> int:
    """Boucle clavier : court-circuite micro et haut-parleur.

    C'est le mode de développement : il prouve la boucle LLM + plugins sans
    qu'une seule ligne d'audio soit nécessaire.
    """
    assistant.pipeline.attach()
    annonceur: asyncio.Task | None = None
    try:
        if once is not None:
            result = await assistant.pipeline.handle_and_speak(once)
            return 0 if (result.skill_result is None or result.skill_result.ok) else 1

        if start_hot_reload(assistant):
            print("Rechargement à chaud actif : déposez un fichier dans plugins/.")
        annonceur = asyncio.ensure_future(assistant.pipeline.run_announcer())

        print(f"Capucine — mode texte ({assistant.llm.describe()}, profil "
              f"{assistant.config.get('profile')}). /aide pour les commandes.")
        if assistant.llm.name == "mock":
            print("Moteur factice : le routage déterministe fonctionne, la conversation "
                  "libre non. Formulez les commandes au plus près des exemples des plugins.")
        _annoncer_demarrage(assistant)

        while True:
            try:
                line = (await asyncio.to_thread(input, "\nVous  › ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line.startswith("/"):
                if _handle_command(assistant, line):
                    return 0
                continue
            await assistant.pipeline.handle_and_speak(line)
    finally:
        if annonceur is not None:
            annonceur.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await annonceur
        assistant.pipeline.detach()


# --- mode vocal ------------------------------------------------------------

class _LecteurClavier:
    """Un seul thread lit l'entrée standard, pour la durée de la session.

    Ouvrir un ``input()`` par attente laisserait des threads bloqués derrière
    soi dès qu'une capture se termine d'elle-même : le thread suivant avalerait
    la frappe destinée au tour d'après.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.lignes: asyncio.Queue[str | None] = asyncio.Queue()
        self._thread = threading.Thread(target=self._lire, name="clavier", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _lire(self) -> None:
        while True:
            try:
                ligne = input()
            except (EOFError, KeyboardInterrupt):
                self.loop.call_soon_threadsafe(self.lignes.put_nowait, None)
                return
            self.loop.call_soon_threadsafe(self.lignes.put_nowait, ligne)

    async def prochaine(self) -> str | None:
        return await self.lignes.get()


async def run_voice_mode(
    assistant: Assistant,
    *,
    once: bool = False,
    push_to_talk: bool = False,
    use_wake: bool = True,
) -> int:
    """Trois façons de parler à Capucine, du plus manuel au plus autonome.

    * ``once`` : un seul tour, depuis un fichier WAV — démonstration et
      intégration continue, sans micro.
    * ``push_to_talk`` : [Entrée] pour parler, [Entrée] pour terminer. Utile
      pour mettre au point sans dépendre du mot d'éveil.
    * défaut : écoute permanente, réveil sur « Capucine », fin d'énoncé au
      VAD, mode suivi, barge-in.
    """
    pipeline = assistant.pipeline
    pipeline.attach()
    annonceur: asyncio.Task | None = None
    try:
        _annoncer_moteurs(assistant)
        _annoncer_demarrage(assistant)
        print("\nChargement des modèles…")
        await pipeline.warmup()

        if once:
            resultat = await pipeline.voice_turn()
            return 0 if resultat.utterance else 1

        if start_hot_reload(assistant):
            print("Rechargement à chaud actif : déposez un fichier dans plugins/.")
        annonceur = asyncio.ensure_future(pipeline.run_announcer())

        if push_to_talk:
            return await _boucle_appui_pour_parler(assistant)
        return await _boucle_continue(assistant, use_wake=use_wake)
    finally:
        if annonceur is not None:
            annonceur.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await annonceur
        pipeline.detach()


def _annoncer_moteurs(assistant: Assistant) -> None:
    print(
        f"Capucine — mode vocal ({assistant.llm.describe()}, "
        f"{assistant.stt.describe() if assistant.stt else 'sans oreille'}, "
        f"{assistant.tts.describe() if assistant.tts else 'sans voix'})."
    )
    if not assistant.pipeline.has_voice:
        print("Aucune voix disponible : les réponses seront affichées.")


async def _boucle_continue(assistant: Assistant, *, use_wake: bool = True) -> int:
    """Écoute permanente : le micro reste ouvert du début à la fin."""
    pipeline = assistant.pipeline
    listener = build_listener(assistant, use_wake=use_wake)
    mot = assistant.config.get("wake.word", "capucine")

    if listener.wake is None:
        use_wake = False
        print("Sans mot d'éveil : je réagis à tout ce que j'entends.")
    else:
        print(f"Dites « {mot} » pour me réveiller.")
    if float(assistant.config.get("assistant.follow_up_seconds", 8.0)) > 0:
        print("Après ma réponse, enchaînez sans redire mon nom.")
    print("Tapez une phrase pour me la dire au clavier. /aide, /quitter.\n")

    listener.start()
    arret = asyncio.Event()
    conversation = asyncio.ensure_future(
        pipeline.run_conversation(listener, use_wake=use_wake, stop=arret)
    )
    clavier = asyncio.ensure_future(_boucle_clavier(assistant, arret))
    try:
        await asyncio.wait({conversation, clavier}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        arret.set()
        listener.stop()
        for tache in (conversation, clavier):
            tache.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tache
    return 0


async def _boucle_clavier(assistant: Assistant, arret: asyncio.Event) -> None:
    """Les commandes restent accessibles pendant que Capucine écoute."""
    clavier = _LecteurClavier(asyncio.get_running_loop())
    clavier.start()
    while not arret.is_set():
        ligne = await clavier.prochaine()
        if ligne is None:
            arret.set()
            return
        ligne = ligne.strip()
        if not ligne:
            continue
        if ligne.startswith("/"):
            if _handle_command(assistant, ligne):
                arret.set()
                return
            continue
        # Une phrase tapée est traitée comme si elle avait été dite.
        await assistant.pipeline.handle_and_speak(ligne)


async def _boucle_appui_pour_parler(assistant: Assistant) -> int:
    """Le déclencheur de l'étape 2, conservé : il ne dépend d'aucun modèle."""
    pipeline = assistant.pipeline
    clavier = _LecteurClavier(asyncio.get_running_loop())
    clavier.start()
    print("[Entrée] pour parler, [Entrée] à nouveau pour terminer. /aide, /quitter.")

    while True:
        print("\n[Entrée] pour parler › ", end="", flush=True)
        ligne = await clavier.prochaine()
        if ligne is None:
            print()
            return 0
        ligne = ligne.strip()
        if ligne.startswith("/"):
            if _handle_command(assistant, ligne):
                return 0
            continue
        if ligne:
            await pipeline.handle_and_speak(ligne)
            continue

        print("… j'écoute, [Entrée] pour terminer.")
        stop = threading.Event()
        tour = asyncio.ensure_future(pipeline.voice_turn(stop))
        fin = asyncio.ensure_future(clavier.prochaine())
        termine, _ = await asyncio.wait({tour, fin}, return_when=asyncio.FIRST_COMPLETED)
        if fin in termine:
            stop.set()
            await tour
        else:
            # La durée maximale a été atteinte : on rend la frappe suivante
            # au tour d'après plutôt que de la perdre.
            fin.cancel()


def describe_startup(assistant: Assistant) -> dict[str, Any]:
    return {
        "profil": assistant.config.get("profile"),
        "llm": assistant.llm.describe(),
        "stt": assistant.stt.describe() if assistant.stt else "-",
        "tts": assistant.tts.describe() if assistant.tts else "-",
        "vad": assistant.vad.describe() if assistant.vad else "-",
        "memoire": "active" if assistant.memoire else "-",
        "atelier": assistant.atelier.decrire() if assistant.atelier else "-",
        "eveil": assistant.wake.describe() if assistant.wake else "-",
        "plugins": len([r for r in assistant.registry.plugins.values() if r.ok]),
        "competences": len(assistant.registry.skills),
        "echecs": len(assistant.registry.failures()),
    }

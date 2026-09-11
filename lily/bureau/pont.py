"""Le pont entre Qt et Lily.

Deux boucles d'événements qui ne se parlent pas nativement : Qt tient la
fenêtre, ``asyncio`` tient le pipeline. Plutôt que de les marier — ce que font
des greffons comme ``qasync``, au prix d'une dépendance et de surprises à
l'arrêt — on garde chacune chez elle :

* un **fil de fond** fait tourner sa propre boucle asyncio et l'assistant ;
* les ordres de l'interface y entrent par une file, via
  ``call_soon_threadsafe`` ;
* ce qui remonte passe par des **signaux Qt**, qui traversent les fils sans
  qu'on ait rien à verrouiller — une connexion entre deux fils est mise en
  file automatiquement.

C'est exactement ce que le pipeline fait déjà entre le fil du micro et sa
boucle. On ne fait qu'étendre le motif d'un cran.

Ce pont ne décide de rien. Il ne connaît ni compétence, ni routage, ni
configuration : il démarre un assistant, lui passe des phrases, et rapporte.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from ..app import Assistant, build_assistant, build_listener
from ..core.config import Config, load_config
from ..core.errors import EngineUnavailable, LilyError
from ..core.logging import get_logger

logger = get_logger("bureau.pont")


@dataclass
class EtatDuDemarrage:
    """Ce que l'interface doit savoir une fois l'assistant debout."""

    llm: str
    stt: str = "-"
    tts: str = "-"
    eveil: str = "-"
    competences: int = 0
    atelier: str = ""
    voix_possible: bool = False
    micro_possible: bool = False
    remarques: list[str] = field(default_factory=list)


class PontLily(QObject):
    """L'assistant, vu depuis la fenêtre."""

    # --- ce qui remonte vers l'interface ---
    prete = Signal(object)        # EtatDuDemarrage
    echouee = Signal(str)
    etat_change = Signal(str)     # l'état du pipeline, en clair
    phrase_dite = Signal(str)     # une phrase part au haut-parleur (ou s'affiche)
    tour_termine = Signal(object) # un TurnResult, d'où qu'il vienne
    annonce = Signal(str)         # un minuteur qui sonne, une compétence nouvelle
    micro_change = Signal(bool)
    voix_change = Signal(bool)
    occupee = Signal(bool)

    def __init__(self, config: Config | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._assistant: Assistant | None = None
        self._boucle: asyncio.AbstractEventLoop | None = None
        self._fil: threading.Thread | None = None
        self._ordres: asyncio.Queue[tuple[str, Any]] | None = None
        self._conversation: asyncio.Task | None = None
        self._annonceur: asyncio.Task | None = None
        self._sortie_audio: Any = None      # mise de côté quand la voix est coupée
        self._micro = False
        self._voix = True

    # -- cycle de vie --------------------------------------------------------
    def demarrer(self) -> None:
        """Lance le fil de fond. Rend la main tout de suite."""
        if self._fil is not None:
            return
        self._fil = threading.Thread(target=self._vivre, name="lily", daemon=True)
        self._fil.start()

    def arreter(self, delai: float = 6.0) -> None:
        self._ordonner("arreter", None)
        if self._fil is not None:
            self._fil.join(delai)
            self._fil = None

    def _vivre(self) -> None:
        """Le point d'entrée du fil de fond : une boucle, un assistant."""
        try:
            asyncio.run(self._servir())
        except Exception as exc:  # pragma: no cover - remonté à l'interface
            logger.exception("Le fil de Lily s'est arrêté.")
            self.echouee.emit(str(exc))

    async def _servir(self) -> None:
        self._boucle = asyncio.get_running_loop()
        self._ordres = asyncio.Queue()
        try:
            etat = await asyncio.to_thread(self._monter)
        except LilyError as exc:
            self.echouee.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover - configuration exotique
            logger.exception("Montage impossible.")
            self.echouee.emit(f"{type(exc).__name__} : {exc}")
            return

        assert self._assistant is not None
        self._assistant.pipeline.attach()
        self._annonceur = asyncio.ensure_future(self._assistant.pipeline.run_announcer())
        self.prete.emit(etat)
        try:
            await self._boucle_des_ordres()
        finally:
            await self._demonter()

    def _monter(self) -> EtatDuDemarrage:
        """Construit l'assistant, en se rabattant sur le texte si l'audio manque.

        C'est le seul endroit du pont qui ait le droit d'être indulgent : une
        fenêtre qui refuse de s'ouvrir parce qu'il manque un micro n'aide
        personne, alors qu'une fenêtre ouverte et franche sur ce qui manque,
        oui.
        """
        config = self._config or load_config()
        self._config = config
        remarques: list[str] = []
        try:
            assistant = build_assistant(config, voice=True)
            audio = True
        except (EngineUnavailable, OSError) as exc:
            remarques.append(f"Chaîne audio indisponible : {exc}")
            logger.warning("Montage sans audio : %s", exc)
            assistant = build_assistant(config, voice=False)
            audio = False

        self._assistant = assistant
        pipeline = assistant.pipeline
        pipeline.echo = False
        pipeline._on_state = lambda etat: self.etat_change.emit(str(etat))
        pipeline._on_phrase = self.phrase_dite.emit
        pipeline._on_turn = self.tour_termine.emit
        # `on_phrase` a déjà porté le texte à la fenêtre. Sans ce muet, le
        # pipeline le réimprimerait dans une console que personne ne regarde —
        # et qui n'existe pas derrière un exécutable sans console.
        pipeline._speak = lambda _texte: None
        self._sortie_audio = pipeline.audio_out

        for remarque in _remarques_de_montage(assistant):
            remarques.append(remarque)

        return EtatDuDemarrage(
            llm=assistant.llm.describe(),
            stt=assistant.stt.describe() if assistant.stt else "-",
            tts=assistant.tts.describe() if assistant.tts else "-",
            eveil=assistant.wake.describe() if assistant.wake else "-",
            competences=len(assistant.registry.skills),
            atelier=str(assistant.atelier.racines[0]) if assistant.atelier.racines else "",
            voix_possible=pipeline.tts is not None and self._sortie_audio is not None,
            micro_possible=audio and assistant.audio_in is not None,
            remarques=remarques,
        )

    async def _demonter(self) -> None:
        await self._couper_le_micro()
        for tache in (self._annonceur,):
            if tache is not None:
                tache.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await tache
        if self._assistant is not None:
            with contextlib.suppress(Exception):
                await self._assistant.aclose()
            self._assistant = None

    # -- les ordres de l'interface -------------------------------------------
    def _ordonner(self, verbe: str, charge: Any) -> None:
        """Dépose un ordre dans la file du fil de fond, depuis n'importe où."""
        boucle, ordres = self._boucle, self._ordres
        if boucle is None or ordres is None or boucle.is_closed():
            logger.debug("Ordre « %s » ignoré : Lily n'est pas debout.", verbe)
            return
        boucle.call_soon_threadsafe(ordres.put_nowait, (verbe, charge))

    @Slot(str)
    def envoyer(self, texte: str) -> None:
        self._ordonner("dire", texte)

    @Slot(bool)
    def regler_micro(self, actif: bool) -> None:
        self._ordonner("micro", bool(actif))

    @Slot(bool)
    def regler_voix(self, active: bool) -> None:
        self._ordonner("voix", bool(active))

    async def _boucle_des_ordres(self) -> None:
        assert self._ordres is not None
        en_cours: asyncio.Task | None = None
        while True:
            verbe, charge = await self._ordres.get()
            if verbe == "arreter":
                if en_cours is not None and not en_cours.done():
                    en_cours.cancel()
                return
            if verbe == "dire":
                # Le tour part dans sa propre tâche : sans cela, couper la voix
                # au milieu d'une réponse attendrait la fin de la réponse.
                en_cours = asyncio.ensure_future(self._un_tour(str(charge)))
            elif verbe == "micro":
                await (self._ouvrir_le_micro() if charge else self._couper_le_micro())
            elif verbe == "voix":
                self._regler_la_voix(bool(charge))

    async def _un_tour(self, texte: str) -> None:
        assert self._assistant is not None
        self.occupee.emit(True)
        try:
            await self._assistant.pipeline.handle_and_speak(texte)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - le pipeline ne lève pas
            logger.exception("Tour en échec.")
        finally:
            self.occupee.emit(False)

    # -- les deux interrupteurs ----------------------------------------------
    def _regler_la_voix(self, active: bool) -> None:
        """Couper la voix, c'est retirer le haut-parleur au pipeline.

        Il sait déjà quoi faire d'une sortie absente : il affiche au lieu de
        dire, par le même chemin que le mode clavier. Rien à inventer.
        """
        assert self._assistant is not None
        pipeline = self._assistant.pipeline
        if active:
            pipeline.audio_out = self._sortie_audio
        else:
            with contextlib.suppress(Exception):
                if pipeline.audio_out is not None:
                    pipeline.audio_out.stop()
            pipeline.audio_out = None
        self._voix = active
        self.voix_change.emit(active)

    async def _ouvrir_le_micro(self) -> None:
        if self._micro or self._assistant is None:
            return
        assistant = self._assistant
        if assistant.audio_in is None:
            self.echouee.emit("Aucune entrée audio : le micro reste fermé.")
            return
        try:
            ecouteur = await asyncio.to_thread(build_listener, assistant)
        except Exception as exc:
            logger.exception("Écoute impossible.")
            self.echouee.emit(f"Écoute impossible : {exc}")
            return
        ecouteur.start()
        self._conversation = asyncio.ensure_future(
            assistant.pipeline.run_conversation(ecouteur, use_wake=assistant.wake is not None)
        )
        self._micro = True
        self.micro_change.emit(True)

    async def _couper_le_micro(self) -> None:
        if not self._micro or self._assistant is None:
            return
        ecouteur = self._assistant.listener
        if ecouteur is not None:
            await asyncio.to_thread(ecouteur.stop)
        if self._conversation is not None:
            self._conversation.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._conversation
            self._conversation = None
        self._micro = False
        self.micro_change.emit(False)

    # -- lecture -------------------------------------------------------------
    @property
    def assistant(self) -> Assistant | None:
        return self._assistant

    @property
    def dossier_de_travail(self) -> Path | None:
        if self._assistant is None or not self._assistant.atelier.racines:
            return None
        return self._assistant.atelier.racines[0]


def _remarques_de_montage(assistant: Assistant) -> list[str]:
    """Ce qui manque, dit une fois, en clair."""
    remarques: list[str] = []
    if assistant.llm.name == "mock":
        remarques.append(
            "Aucun modèle de langage joignable : les compétences répondent, "
            "la conversation libre non. Vérifiez qu'Ollama tourne."
        )
    if assistant.tts is None:
        remarques.append("Aucune voix : les réponses seront affichées.")
    if assistant.wake is None and assistant.audio_in is not None:
        remarques.append(
            "Aucun mot d'éveil : le micro écoutera tout ce qu'il entend."
        )
    for record in assistant.registry.failures():
        remarques.append(f"Plugin ignoré — {record.name} : {record.error}")
    return remarques

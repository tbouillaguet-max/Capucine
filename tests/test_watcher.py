"""Rechargement à chaud : déposer un fichier suffit."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from lily.core.registry import PluginRegistry
from lily.core.watcher import ChangeKind, PluginWatcher, traduire_evenement

PLUGIN = '''
from lily.plugin import skill

@skill(description="Dit bonjour.", examples=["dis bonjour"])
def saluer() -> str:
    return "Bonjour."
'''


@pytest.fixture
def surveillance(dossier_plugins: Path, tmp_path: Path):
    registry = PluginRegistry([dossier_plugins], data_root=tmp_path / "data")
    registry.load_all()
    watcher = PluginWatcher(registry, [dossier_plugins], debounce_ms=0)
    yield watcher, registry, dossier_plugins
    for nom in list(registry.plugins):
        registry.unload(nom, notify=False)


def test_un_fichier_depose_devient_une_competence(surveillance, ecrire_plugin) -> None:
    watcher, registry, _ = surveillance
    assert registry.skills == {}

    chemin = ecrire_plugin("salutations.py", PLUGIN)
    watcher.notify(chemin, ChangeKind.UPSERT)
    watcher.flush(force=True)

    assert "saluer" in registry.skills
    assert registry.call("saluer").speak == "Bonjour."


def test_une_rafale_d_evenements_ne_donne_qu_un_rechargement(surveillance, ecrire_plugin) -> None:
    # Un éditeur produit trois ou quatre événements par sauvegarde : écriture
    # d'un temporaire, renommage, changement de droits.
    watcher, _, _ = surveillance
    chemin = ecrire_plugin("salutations.py", PLUGIN)
    for _ in range(4):
        watcher.notify(chemin, ChangeKind.UPSERT)
    watcher.flush(force=True)
    assert watcher.applications == 1


def test_une_reecriture_a_l_identique_ne_recharge_pas(surveillance, ecrire_plugin) -> None:
    # Un formateur, un `touch`, une synchronisation : le contenu n'a pas
    # changé, il n'y a rien à recharger et rien à annoncer.
    watcher, _, _ = surveillance
    chemin = ecrire_plugin("salutations.py", PLUGIN)
    watcher.notify(chemin, ChangeKind.UPSERT)
    watcher.flush(force=True)
    assert watcher.applications == 1

    chemin.write_text(PLUGIN, encoding="utf-8")   # même octets
    watcher.notify(chemin, ChangeKind.UPSERT)
    watcher.flush(force=True)
    assert watcher.applications == 1


def test_une_modification_reelle_recharge(surveillance, ecrire_plugin) -> None:
    watcher, registry, _ = surveillance
    chemin = ecrire_plugin("salutations.py", PLUGIN)
    watcher.notify(chemin, ChangeKind.UPSERT)
    watcher.flush(force=True)

    chemin.write_text(PLUGIN.replace("Bonjour.", "Bonsoir."), encoding="utf-8")
    watcher.notify(chemin, ChangeKind.UPSERT)
    watcher.flush(force=True)
    assert registry.call("saluer").speak == "Bonsoir."


def test_une_modification_de_meme_longueur_est_bien_prise(surveillance, ecrire_plugin) -> None:
    # Le piège : CPython valide son cache de bytecode sur (date, taille).
    # « Bonjour. » et « Bonsoir. » font la même longueur, et deux sauvegardes
    # dans la même seconde ont la même date : l'ancien code serait rejoué.
    watcher, registry, _ = surveillance
    chemin = ecrire_plugin("salutations.py", PLUGIN)
    watcher.notify(chemin, ChangeKind.UPSERT)
    watcher.flush(force=True)
    assert registry.call("saluer").speak == "Bonjour."

    chemin.write_text(PLUGIN.replace("Bonjour.", "Bonsoir."), encoding="utf-8")
    watcher.notify(chemin, ChangeKind.UPSERT)
    watcher.flush(force=True)
    assert registry.call("saluer").speak == "Bonsoir."


def test_un_fichier_supprime_retire_ses_competences(surveillance, ecrire_plugin) -> None:
    watcher, registry, _ = surveillance
    chemin = ecrire_plugin("salutations.py", PLUGIN)
    watcher.notify(chemin, ChangeKind.UPSERT)
    watcher.flush(force=True)
    assert "saluer" in registry.skills

    chemin.unlink()
    watcher.notify(chemin, ChangeKind.DELETE)
    watcher.flush(force=True)
    assert "saluer" not in registry.skills


def test_un_renommage_deplace_la_competence(surveillance, ecrire_plugin, dossier_plugins) -> None:
    watcher, registry, _ = surveillance
    ancien = ecrire_plugin("ancien.py", PLUGIN)
    watcher.notify(ancien, ChangeKind.UPSERT)
    watcher.flush(force=True)
    assert registry.plugins["ancien"].ok

    nouveau = dossier_plugins / "nouveau.py"
    ancien.rename(nouveau)
    traduire_evenement(watcher, _Evenement("moved", str(ancien), str(nouveau)))
    watcher.flush(force=True)

    assert "ancien" not in registry.plugins
    assert registry.plugins["nouveau"].ok


def test_un_plugin_casse_puis_reparé_finit_par_marcher(surveillance, ecrire_plugin) -> None:
    watcher, registry, _ = surveillance
    chemin = ecrire_plugin("bancal.py", "def f(:\n")
    watcher.notify(chemin, ChangeKind.UPSERT)
    watcher.flush(force=True)
    assert [e.name for e in registry.failures()] == ["bancal"]

    chemin.write_text(PLUGIN, encoding="utf-8")
    watcher.notify(chemin, ChangeKind.UPSERT)
    watcher.flush(force=True)
    assert not registry.failures()
    assert "saluer" in registry.skills


@pytest.mark.parametrize("nom", ["notes.txt", "_prive.py", ".cache.py", "README.md"])
def test_les_fichiers_hors_sujet_sont_ignores(surveillance, dossier_plugins, nom) -> None:
    watcher, _, _ = surveillance
    (dossier_plugins / nom).write_text("x", encoding="utf-8")
    watcher.notify(dossier_plugins / nom, ChangeKind.UPSERT)
    assert watcher.flush(force=True) == []


def test_un_fichier_hors_des_dossiers_surveilles_est_ignore(surveillance, tmp_path) -> None:
    watcher, _, _ = surveillance
    ailleurs = tmp_path / "ailleurs.py"
    ailleurs.write_text(PLUGIN, encoding="utf-8")
    watcher.notify(ailleurs, ChangeKind.UPSERT)
    assert watcher.flush(force=True) == []


def test_l_anti_rebond_attend_la_fin_de_la_rafale(dossier_plugins, tmp_path) -> None:
    registry = PluginRegistry([dossier_plugins], data_root=tmp_path / "data")
    watcher = PluginWatcher(registry, [dossier_plugins], debounce_ms=200)
    chemin = dossier_plugins / "salutations.py"
    chemin.write_text(PLUGIN, encoding="utf-8")

    watcher.notify(chemin, ChangeKind.UPSERT)
    assert watcher.flush() == []          # trop tôt
    time.sleep(0.25)
    assert watcher.flush() == [chemin]    # la rafale est terminée


@dataclass
class _Evenement:
    event_type: str
    src_path: str = ""
    dest_path: str = ""
    is_directory: bool = False


def test_traduction_des_evenements_watchdog(surveillance, dossier_plugins) -> None:
    watcher, _, _ = surveillance
    chemin = dossier_plugins / "a.py"

    for type_evenement in ("created", "modified", "closed"):
        traduire_evenement(watcher, _Evenement(type_evenement, str(chemin)))
        assert list(watcher._en_attente) == [chemin]
        watcher._en_attente.clear()

    traduire_evenement(watcher, _Evenement("deleted", str(chemin)))
    assert watcher._en_attente[chemin].kind is ChangeKind.DELETE

    watcher._en_attente.clear()
    traduire_evenement(watcher, _Evenement("modified", str(dossier_plugins), is_directory=True))
    assert watcher._en_attente == {}


def test_sans_watchdog_on_le_dit_et_on_continue(surveillance, monkeypatch) -> None:
    watcher, _, _ = surveillance
    # On simule l'absence du paquet : le démarrage échoue proprement, sans
    # exception, et /recharge reste disponible.
    monkeypatch.setitem(sys.modules, "watchdog.observers", None)
    assert watcher.start() is False
    assert not watcher.active


def test_l_observateur_reel_voit_un_fichier_depose(dossier_plugins, tmp_path) -> None:
    """Le vrai chemin : watchdog, un vrai fichier, un vrai dossier."""
    pytest.importorskip("watchdog", reason="watchdog n'est pas installé")

    registry = PluginRegistry([dossier_plugins], data_root=tmp_path / "data")
    registry.load_all()
    watcher = PluginWatcher(registry, [dossier_plugins], debounce_ms=100, poll_ms=20)
    assert watcher.start()
    try:
        (dossier_plugins / "tardif.py").write_text(PLUGIN, encoding="utf-8")
        limite = time.monotonic() + 10
        while "saluer" not in registry.skills and time.monotonic() < limite:
            time.sleep(0.05)
        assert "saluer" in registry.skills, "le fichier déposé n'a pas été vu"
    finally:
        watcher.stop()
        for nom in list(registry.plugins):
            registry.unload(nom, notify=False)

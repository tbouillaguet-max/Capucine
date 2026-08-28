"""Agrégation des latences : où part la seconde et demie."""

from __future__ import annotations

import pytest

from capucine.core.logging import LatencyBook, TurnTelemetry, get_latency_book


@pytest.fixture
def carnet() -> LatencyBook:
    livre = get_latency_book()
    livre.reset()
    yield livre
    livre.reset()


def test_un_carnet_vide_le_dit() -> None:
    livre = LatencyBook()
    assert livre.snapshot() == []
    assert "Aucune latence" in livre.table()


def test_les_centiles_sont_justes() -> None:
    livre = LatencyBook()
    for valeur in range(1, 11):        # 1 à 10 ms
        livre.record("transcription", valeur)
    (stat,) = livre.snapshot()
    assert stat.nombre == 10
    assert stat.p50_ms == pytest.approx(5.5, abs=0.6)
    assert stat.p90_ms == pytest.approx(9.0, abs=1.0)
    assert stat.max_ms == 10.0


def test_l_etage_le_plus_couteux_vient_en_premier() -> None:
    livre = LatencyBook()
    livre.record("routage", 5)
    livre.record("transcription", 900)
    livre.record("synthese", 200)
    assert [s.etage for s in livre.snapshot()] == ["transcription", "synthese", "routage"]


def test_les_echantillons_sont_bornes() -> None:
    # Un assistant qui tourne des semaines ne doit pas grossir indéfiniment.
    livre = LatencyBook(max_samples=50)
    for valeur in range(500):
        livre.record("routage", valeur)
    (stat,) = livre.snapshot()
    assert stat.nombre == 50
    assert stat.max_ms == 499.0      # ce sont bien les plus récents


def test_le_tableau_choisit_son_unite() -> None:
    livre = LatencyBook()
    livre.record("transcription", 1850)
    livre.record("routage", 3.2)
    tableau = livre.table()
    assert "1.85 s" in tableau       # une seconde et demie ne se lit pas en ms
    assert "3.2 ms" in tableau


def test_un_tour_alimente_le_carnet(carnet: LatencyBook) -> None:
    telemetrie = TurnTelemetry(name="tour")
    telemetrie.record("transcription_ms", 900)
    telemetrie.record("reflexion_ms", 40)
    telemetrie.emit(etage="regle")

    etages = {stat.etage for stat in carnet.snapshot()}
    # Le suffixe technique disparaît, et le total est ajouté.
    assert {"transcription", "reflexion", "total"} <= etages
    assert "transcription_ms" not in etages


def test_la_remise_a_zero_vide_tout(carnet: LatencyBook) -> None:
    carnet.record("routage", 10)
    assert carnet.snapshot()
    carnet.reset()
    assert carnet.snapshot() == []


# --- le banc de mesure ------------------------------------------------------

def _charger_outil():
    """Importe tools/mesurer_latence.py, qui n'est pas un paquet."""
    import importlib.util
    import sys

    from capucine.core.config import PROJECT_ROOT

    chemin = PROJECT_ROOT / "tools" / "mesurer_latence.py"
    spec = importlib.util.spec_from_file_location("mesurer_latence", chemin)
    module = importlib.util.module_from_spec(spec)
    # Nécessaire avant l'exécution : les dataclasses du module résolvent leurs
    # annotations via sys.modules[cls.__module__].
    sys.modules["mesurer_latence"] = module
    spec.loader.exec_module(module)
    return module


def test_le_banc_de_mesure_rend_du_json_exploitable(capsys, tmp_path) -> None:
    # Ce fichier est un livrable : s'il pourrit, on ne s'en aperçoit qu'au
    # moment où l'on cherche à comprendre pourquoi le Pi rame.
    import json

    outil = _charger_outil()
    fichier = tmp_path / "essai.toml"
    fichier.write_text(
        '[llm]\nengine = "mock"\n[stt]\nengine = "vosk"\n'
        f'model_path = "{tmp_path / "absent"}"\n'
        '[tts]\nengine = "piper"\n'
        f'models_dir = "{tmp_path / "absent"}"\n',
        encoding="utf-8",
    )
    code = outil.main(["--config", str(fichier), "--repetitions", "6", "--json"])
    assert code == 0

    rapport = json.loads(capsys.readouterr().out)
    assert rapport["machine"]["coeurs"] >= 1
    etages = {mesure["etage"] for mesure in rapport["mesures"]}
    assert {"mot d'éveil", "VAD", "transcription", "routage déterministe"} <= etages
    # Ce qui n'est pas installé est marqué indisponible, avec la raison — le
    # banc ne fabrique jamais de chiffre.
    for mesure in rapport["mesures"]:
        assert mesure["disponible"] or mesure["note"]
        if mesure["disponible"]:
            assert mesure["p50_ms"] is not None


def test_le_banc_de_mesure_rend_un_rapport_lisible(capsys, tmp_path) -> None:
    outil = _charger_outil()
    fichier = tmp_path / "essai.toml"
    fichier.write_text('[llm]\nengine = "mock"\n', encoding="utf-8")

    assert outil.main(["--config", str(fichier), "--repetitions", "6"]) == 0
    sortie = capsys.readouterr().out
    assert "Étages permanents" in sortie
    assert "Étages à la demande" in sortie
    assert "temps réel" in sortie

"""Configuration en couches."""

from __future__ import annotations

from pathlib import Path

import pytest

from capucine.core.config import Config, detect_profile, load_config
from capucine.core.errors import ConfigError


@pytest.fixture
def dossier_config(tmp_path: Path) -> Path:
    dossier = tmp_path / "config"
    dossier.mkdir()
    (dossier / "default.toml").write_text(
        '[llm]\nengine = "mock"\nmodel = "petit"\ntemperature = 0.3\n'
        '[llm.router]\ndirect_threshold = 0.72\n'
        '[plugins]\ntimeout = 10.0\npaths = ["./plugins"]\n',
        encoding="utf-8",
    )
    (dossier / "pi.toml").write_text('[llm]\nmodel = "minuscule"\n', encoding="utf-8")
    return dossier


def test_le_profil_surcharge_les_defauts(dossier_config: Path) -> None:
    config = load_config(profile="pi", environ={}, config_dir=dossier_config)
    assert config.get("llm.model") == "minuscule"
    # La fusion est profonde : ce que le profil ne dit pas est conservé.
    assert config.get("llm.engine") == "mock"
    assert config.get("llm.router.direct_threshold") == 0.72
    assert config.get("profile") == "pi"


def test_l_environnement_surcharge_le_profil(dossier_config: Path) -> None:
    config = load_config(
        profile="pi",
        environ={"CAPUCINE_LLM__MODEL": "depuis-env", "CAPUCINE_PLUGINS__TIMEOUT": "2.5"},
        config_dir=dossier_config,
    )
    assert config.get("llm.model") == "depuis-env"
    assert config.get("plugins.timeout") == 2.5


def test_la_ligne_de_commande_surcharge_tout(dossier_config: Path) -> None:
    config = load_config(
        profile="pi",
        environ={"CAPUCINE_LLM__MODEL": "depuis-env"},
        overrides={"llm": {"model": "depuis-cli"}},
        config_dir=dossier_config,
    )
    assert config.get("llm.model") == "depuis-cli"


def test_les_valeurs_d_environnement_sont_typees(dossier_config: Path) -> None:
    config = load_config(
        environ={
            "CAPUCINE_A__ENTIER": "12",
            "CAPUCINE_A__FLOTTANT": "0.5",
            "CAPUCINE_A__VRAI": "oui",
            "CAPUCINE_A__FAUX": "false",
            "CAPUCINE_A__LISTE": "un, deux",
            "CAPUCINE_A__TEXTE": "bonjour",
        },
        config_dir=dossier_config,
    )
    assert config.get("a.entier") == 12
    assert config.get("a.flottant") == 0.5
    assert config.get("a.vrai") is True
    assert config.get("a.faux") is False
    assert config.get("a.liste") == ["un", "deux"]
    assert config.get("a.texte") == "bonjour"


def test_config_de_plugin_fusionne_les_defauts_du_module() -> None:
    config = Config({"plugins": {"calcul": {"decimales": 4}}})
    fusion = config.plugin_config("calcul", {"decimales": 2, "autre": True})
    assert fusion == {"decimales": 4, "autre": True}
    # Un plugin sans section garde ses propres défauts.
    assert config.plugin_config("inconnu", {"a": 1}) == {"a": 1}


def test_un_plugin_ne_peut_pas_ecraser_une_cle_du_coeur() -> None:
    # Les plugins vivent sous `plugins.`, jamais à la racine : un plugin
    # nommé « llm » ne touche pas à la configuration du moteur.
    config = Config({"llm": {"engine": "ollama"}, "plugins": {"llm": {"engine": "n'importe quoi"}}})
    assert config.get("llm.engine") == "ollama"
    assert config.plugin_config("llm") == {"engine": "n'importe quoi"}


def test_fichier_absent_donne_une_erreur_lisible(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="introuvable"):
        load_config(environ={}, config_dir=tmp_path / "nulle-part")


def test_le_profil_est_detecte() -> None:
    assert detect_profile() in ("pc", "pi")


def test_la_config_livree_est_valide() -> None:
    # Filet contre une faute de frappe dans config/default.toml.
    config = load_config(profile="pc", environ={})
    assert config.get("assistant.name") == "Capucine"
    assert config.get("llm.engine")
    assert config.plugin_paths()

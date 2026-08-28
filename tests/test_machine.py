"""Ce que Capucine sait de la machine, et ce qu'elle en conclut."""

from __future__ import annotations

import pytest

from capucine.core.config import Config, load_config
from capucine.core.machine import Machine, conseils, decrire


def test_decrire_ne_leve_jamais_et_dit_quelque_chose() -> None:
    machine = decrire()
    assert machine.systeme
    assert machine.coeurs >= 1
    assert machine.profil_conseille in ("pc", "pi")
    assert machine.resume()


@pytest.mark.parametrize(
    "systeme,architecture,attendu",
    [
        ("Linux", "aarch64", "pi"),
        ("Linux", "armv7l", "pi"),
        ("Linux", "x86_64", "pc"),
        ("Windows", "AMD64", "pc"),
        ("Darwin", "arm64", "pc"),   # un Mac ARM reste une machine principale
    ],
)
def test_le_profil_se_deduit_de_l_architecture(systeme, architecture, attendu) -> None:
    machine = Machine(systeme=systeme, architecture=architecture, coeurs=4)
    assert machine.profil_conseille == attendu


def test_un_raspberry_pi_est_reconnu_par_son_modele() -> None:
    machine = Machine(
        systeme="Linux", architecture="aarch64", coeurs=4,
        modele="Raspberry Pi 4 Model B Rev 1.4", est_pi=True, memoire_go=4.0,
    )
    assert machine.profil_conseille == "pi"
    assert "Raspberry Pi 4" in machine.resume()


# --- conseils ---------------------------------------------------------------

PI = Machine(
    systeme="Linux", architecture="aarch64", coeurs=4,
    memoire_go=2.0, modele="Raspberry Pi 4 Model B", est_pi=True,
)
PC = Machine(systeme="Linux", architecture="x86_64", coeurs=16, memoire_go=32.0)


def test_un_modele_trop_gros_pour_la_memoire_est_signale() -> None:
    config = Config({"profile": "pi", "stt": {"model": "medium"}})
    remarques = " ".join(conseils(config, PI))
    # Le point important : ça ne plantera pas, ce sera lent — et on propose
    # une porte de sortie.
    assert "medium" in remarques and "lent" in remarques
    assert "vosk" in remarques


def test_un_petit_modele_sur_un_pi_ne_declenche_rien() -> None:
    config = Config({
        "profile": "pi",
        "stt": {"model": "tiny"},
        "barge_in": {"mode": "eveil"},
        "llm": {"num_ctx": 2048},
    })
    assert conseils(config, PI) == []


def test_un_profil_qui_ne_correspond_pas_a_la_machine_est_signale() -> None:
    config = Config({"profile": "pi", "stt": {"model": "small"}})
    assert any("--profile pc" in remarque for remarque in conseils(config, PC))


def test_cuda_sans_gpu_est_signale() -> None:
    config = Config({"profile": "pc", "stt": {"model": "small", "device": "cuda"}})
    assert any("cuda" in remarque.lower() for remarque in conseils(config, PC))


def test_le_barge_in_par_la_voix_est_deconseille_sur_un_pi() -> None:
    # Haut-parleur ouvert, pas d'annulation d'écho : elle se couperait elle-même.
    config = Config({"profile": "pi", "stt": {"model": "tiny"}, "barge_in": {"mode": "voix"}})
    assert any("écho" in remarque for remarque in conseils(config, PI))


def test_un_contexte_trop_grand_sur_un_pi_est_signale() -> None:
    config = Config({
        "profile": "pi", "stt": {"model": "tiny"},
        "barge_in": {"mode": "eveil"}, "llm": {"num_ctx": 8192},
    })
    assert any("num_ctx" in remarque for remarque in conseils(config, PI))


def test_peu_de_coeurs_et_un_gros_modele_sont_signales() -> None:
    petite = Machine(systeme="Linux", architecture="aarch64", coeurs=2, memoire_go=8.0)
    config = Config({"profile": "pi", "stt": {"model": "small"}})
    assert any("cœur" in remarque for remarque in conseils(config, petite))


# --- le profil livré --------------------------------------------------------

def test_le_profil_pi_livre_est_coherent() -> None:
    config = load_config(profile="pi", environ={})
    # Ce que le profil doit garantir : rien de trop gros pour une carte.
    assert config.get("stt.model") in ("tiny", "base")
    assert config.get("stt.compute_type") == "int8"
    assert config.get("stt.beam_size") == 1
    assert config.get("llm.num_ctx") <= 2048
    assert config.get("barge_in.mode") == "eveil"
    assert config.get("wake.engine") == "vosk"
    # Et il ne déclenche aucune remarque sur une carte modeste.
    assert conseils(config, PI) == []


def test_le_profil_pc_livre_ne_gene_pas_une_grosse_machine() -> None:
    config = load_config(profile="pc", environ={})
    assert conseils(config, PC) == []

"""Découpage en phrases : l'unité de la synthèse et de l'interruption."""

from __future__ import annotations

import pytest

from lily.core.text import split_sentences, stream_sentences


@pytest.mark.parametrize(
    "texte,attendu",
    [
        ("Il est midi. Le soleil brille.", ["Il est midi.", "Le soleil brille."]),
        ("Vraiment ? Oui ! Enfin...", ["Vraiment ?", "Oui !", "Enfin..."]),
        ("Sans point final", ["Sans point final"]),
        ("", []),
        ("   ", []),
    ],
)
def test_decoupage_simple(texte, attendu) -> None:
    assert split_sentences(texte) == attendu


def test_on_ne_coupe_ni_les_abreviations_ni_les_decimaux() -> None:
    assert split_sentences("M. Dupont a payé 3.5 euros. Il est parti.") == [
        "M. Dupont a payé 3.5 euros.",
        "Il est parti.",
    ]
    # « M. » introduit toujours un nom : on ne coupe jamais après. « etc. »
    # termine le plus souvent une phrase : on coupe.
    assert split_sentences("Des pommes, etc. Puis des poires.") == [
        "Des pommes, etc.",
        "Puis des poires.",
    ]
    assert split_sentences("Mme Martin est là.") == ["Mme Martin est là."]


def test_la_ponctuation_fermante_reste_avec_sa_phrase() -> None:
    assert split_sentences('Il a dit « bonjour ». Puis il est parti.') == [
        "Il a dit « bonjour ».",
        "Puis il est parti.",
    ]


def test_les_fragments_trop_courts_sont_recolles() -> None:
    # Vers l'avant quand il n'y a rien avant, vers l'arrière sinon.
    assert split_sentences("Ah. Il est midi.", min_chars=6) == ["Ah. Il est midi."]
    assert split_sentences("Il est midi. Ah.", min_chars=6) == ["Il est midi. Ah."]


def test_le_flux_ne_perd_aucune_espace() -> None:
    # Le piège : découper puis recoller des morceaux déjà nettoyés soude les
    # mots entre eux.
    morceaux = ["Il est ", "midi. ", "Le soleil ", "brille. Et ", "puis voilà"]
    assert list(stream_sentences(morceaux)) == [
        "Il est midi.", "Le soleil brille.", "Et puis voilà",
    ]


def test_le_flux_rend_une_phrase_des_qu_elle_est_complete() -> None:
    # C'est tout l'intérêt : parler pendant que le modèle écrit la suite.
    rendues: list[str] = []

    def fragments():
        for morceau in ["Il est midi.", " Le soleil", " brille."]:
            yield morceau
            rendues.append(f"<{len(rendues)}>")

    phrases = list(stream_sentences(fragments()))
    assert phrases == ["Il est midi.", "Le soleil brille."]


def test_le_flux_rend_toujours_le_reste_sans_ponctuation() -> None:
    assert list(stream_sentences(["Bonjour", " tout", " le monde"])) == ["Bonjour tout le monde"]


def test_le_flux_ignore_les_fragments_vides() -> None:
    assert list(stream_sentences(["", "Bonjour.", "", ""])) == ["Bonjour."]

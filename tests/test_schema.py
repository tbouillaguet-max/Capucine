"""Génération de schéma d'outil et coercition des arguments.

Le contrat dit : « le développeur du plugin n'écrit jamais de JSON ». Ces
tests vérifient que la signature suffit.
"""

from __future__ import annotations

from typing import Annotated, Literal

import pytest

from lily.core.errors import ArgumentError, SchemaError
from lily.core.schema import (
    build_parameters_schema,
    build_tool_schema,
    coerce_arguments,
    parse_docstring,
)
from lily.core.text import extract_numbers, parse_french_number


def test_le_schema_vient_de_la_signature() -> None:
    def meteo(ville: str, jour: str = "aujourd'hui", detaille: bool = False) -> str:
        """Donne la météo."""

    schema, specs = build_parameters_schema(meteo)
    assert schema["properties"]["ville"] == {"type": "string"}
    assert schema["properties"]["jour"]["type"] == "string"
    assert schema["properties"]["jour"]["default"] == "aujourd'hui"
    assert schema["properties"]["detaille"]["type"] == "boolean"
    # Seul l'argument sans valeur par défaut est obligatoire.
    assert schema["required"] == ["ville"]
    assert schema["additionalProperties"] is False
    assert [s.name for s in specs] == ["ville", "jour", "detaille"]


def test_literal_devient_une_enumeration() -> None:
    def f(jour: Literal["aujourd'hui", "demain"] = "demain"): ...

    schema, _ = build_parameters_schema(f)
    assert schema["properties"]["jour"]["enum"] == ["aujourd'hui", "demain"]


def test_annotated_porte_la_description() -> None:
    def f(volume: Annotated[int, "de 0 à 100"] = 50): ...

    schema, _ = build_parameters_schema(f)
    assert schema["properties"]["volume"]["description"] == "de 0 à 100"


def test_optionnel_et_listes() -> None:
    def f(ville: str | None = None, tags: list[str] | None = None, brut: dict | None = None): ...

    schema, _ = build_parameters_schema(f)
    assert schema["properties"]["ville"]["nullable"] is True
    assert schema["properties"]["tags"]["type"] == "array"
    assert schema["properties"]["tags"]["items"] == {"type": "string"}
    assert schema["properties"]["brut"]["type"] == "object"


def test_sans_annotation_on_devine_depuis_le_defaut() -> None:
    def f(n=3, texte="a"): ...

    schema, _ = build_parameters_schema(f)
    assert schema["properties"]["n"]["type"] == "integer"
    assert schema["properties"]["texte"]["type"] == "string"


def test_args_et_kwargs_sont_refuses() -> None:
    def f(*args, **kwargs): ...

    with pytest.raises(SchemaError, match=r"\*args"):
        build_parameters_schema(f)


class TypeExotique:
    """Défini au niveau du module pour que l'annotation soit résoluble."""


def test_type_non_supporte_donne_une_erreur_explicite() -> None:
    def f(x: TypeExotique): ...

    with pytest.raises(SchemaError, match="non pris en charge"):
        build_parameters_schema(f)


def test_les_annotations_differees_sont_resolues() -> None:
    # Ce fichier commence par « from __future__ import annotations » : les
    # annotations arrivent donc sous forme de chaînes. Un plugin réel fera
    # souvent pareil, et le schéma doit rester juste.
    def f(n: int = 3, actif: bool = False): ...

    assert f.__annotations__["n"] == "int"  # bien une chaîne
    schema, _ = build_parameters_schema(f)
    assert schema["properties"]["n"]["type"] == "integer"
    assert schema["properties"]["actif"]["type"] == "boolean"


def test_une_annotation_irresoluble_degrade_sans_casser() -> None:
    def f(x: TypeQuiNExistePas = "a", n: int = 1): ...  # noqa: F821

    schema, _ = build_parameters_schema(f)
    # L'argument illisible retombe sur « chaîne », les autres restent typés.
    assert schema["properties"]["x"]["type"] == "string"
    assert schema["properties"]["n"]["type"] == "integer"


def test_docstring_rest_et_google() -> None:
    resume, params = parse_docstring(
        """Résumé sur une ligne.

        :param ville: La ville visée.
        """
    )
    assert resume.startswith("Résumé")
    assert params["ville"] == "La ville visée."

    resume, params = parse_docstring(
        """Résumé.

        Args:
            ville: La ville visée.
            jour: Le jour voulu.
        """
    )
    assert params == {"ville": "La ville visée.", "jour": "Le jour voulu."}
    assert "Args:" not in resume


def test_le_schema_d_outil_agrege_description_docstring_et_exemples() -> None:
    def meteo(ville: str):
        """Interroge la station météo locale."""

    schema = build_tool_schema(
        meteo, name="meteo", description="Donne la météo.", examples=["quel temps fait-il"]
    )
    description = schema["function"]["description"]
    assert schema["function"]["name"] == "meteo"
    assert "Donne la météo." in description
    assert "station météo locale" in description
    assert "quel temps fait-il" in description
    assert schema["function"]["parameters"]["required"] == ["ville"]


# --- coercition ------------------------------------------------------------

def test_un_nombre_en_toutes_lettres_est_accepte() -> None:
    def lancer_de(faces: int = 6): ...

    schema, _ = build_parameters_schema(lancer_de)
    propres, _ = coerce_arguments(schema, {"faces": "vingt"})
    assert propres == {"faces": 20}


def test_les_arguments_inventes_sont_ecartes_pas_fatals() -> None:
    def f(a: int = 1): ...

    schema, _ = build_parameters_schema(f)
    propres, ignores = coerce_arguments(schema, {"a": 2, "invente": "x"})
    assert propres == {"a": 2}
    assert ignores == ["invente"]


def test_un_argument_obligatoire_manquant_est_signale() -> None:
    def f(ville: str): ...

    schema, _ = build_parameters_schema(f)
    with pytest.raises(ArgumentError, match="ville"):
        coerce_arguments(schema, {})


@pytest.mark.parametrize("brut,attendu", [("oui", True), ("vrai", True), ("non", False), (False, False)])
def test_les_booleens_acceptent_le_francais(brut, attendu) -> None:
    def f(actif: bool = False): ...

    schema, _ = build_parameters_schema(f)
    propres, _ = coerce_arguments(schema, {"actif": brut})
    assert propres["actif"] is attendu


def test_les_enumerations_tolerent_accents_et_casse() -> None:
    def f(jour: Literal["aujourd'hui", "demain"] = "demain"): ...

    schema, _ = build_parameters_schema(f)
    propres, _ = coerce_arguments(schema, {"jour": "Demain"})
    assert propres["jour"] == "demain"

    with pytest.raises(ArgumentError, match="doit valoir"):
        coerce_arguments(schema, {"jour": "après-demain"})


def test_un_entier_refuse_un_decimal() -> None:
    def f(n: int = 1): ...

    schema, _ = build_parameters_schema(f)
    with pytest.raises(ArgumentError, match="entier"):
        coerce_arguments(schema, {"n": 2.5})


def test_une_liste_accepte_une_chaine_separee_par_virgules() -> None:
    def f(tags: list[str] | None = None): ...

    schema, _ = build_parameters_schema(f)
    propres, _ = coerce_arguments(schema, {"tags": "un, deux"})
    assert propres["tags"] == ["un", "deux"]


# --- nombres français ------------------------------------------------------

@pytest.mark.parametrize(
    "texte,attendu",
    [
        ("vingt", 20), ("quatre-vingt-dix", 90), ("soixante-quinze", 75),
        ("cent vingt", 120), ("deux mille vingt", 2020), ("12", 12),
        ("3,5", 3.5), ("bonjour", None),
    ],
)
def test_lecture_des_nombres_en_toutes_lettres(texte, attendu) -> None:
    assert parse_french_number(texte) == attendu


def test_un_article_n_est_pas_un_nombre() -> None:
    # « lance un dé à vingt faces » ne contient qu'un seul nombre : sans cette
    # règle, l'extraction déterministe du routeur choisirait « un ».
    assert extract_numbers("lance un dé à vingt faces") == [20]
    assert extract_numbers("lance un dé") == []
    assert extract_numbers("minuteur de 5 minutes") == [5]

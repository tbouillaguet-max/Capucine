"""Routage à trois étages.

L'enjeu : ne pas dépendre de la bonne volonté d'un modèle 7-8B quantifié pour
choisir un outil. L'étage déterministe doit trancher les cas nets tout seul,
et l'étage LLM doit être structurellement incapable d'inventer un nom d'outil.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from lily.core.engines.llm.mock import MockLLM
from lily.core.plugin import SkillDeclaration, build_skill_spec
from lily.core.router import NO_TOOL, Router
from lily.core.text import PhrasePreparee, similarity, similarity_preparee


def fabriquer(func, description="", examples=()):
    declaration = SkillDeclaration(
        description=description, examples=tuple(examples), name=None, timeout=None, confirm=False
    )
    return build_skill_spec(
        func, declaration, module="lily.plugins.test", source=Path("test.py"), plugin="test"
    )


def lancer_de(faces: int = 6) -> str:
    """Lance un dé."""
    return "4"


def heure() -> str:
    """Donne l'heure."""
    return "midi"


def repete(texte: str) -> str:
    """Répète un texte."""
    return texte


def minuteur(minutes: int = 5, secondes: int = 0) -> str:
    """Lance un minuteur."""
    return "ok"


DES = fabriquer(lancer_de, "Lance un dé.", ["lance un dé", "lance un dé à six faces"])
HEURE = fabriquer(heure, "Donne l'heure qu'il est.", ["quelle heure est-il"])
ECHO = fabriquer(repete, "Répète un texte.", ["répète après moi"])
MINUTEUR = fabriquer(minuteur, "Lance un minuteur.", ["mets un minuteur"])

CATALOGUE = {s.name: s for s in (DES, HEURE, ECHO, MINUTEUR)}


def test_le_critere_d_acceptation_ne_sollicite_pas_le_modele() -> None:
    # « Lily, lance un dé à vingt faces » doit marcher sans que le LLM
    # ait son mot à dire : c'est ce qui rend la démonstration fiable.
    llm = MockLLM()
    routeur = Router(llm)
    decision = routeur.route("lance un dé à vingt faces", CATALOGUE)

    assert decision.tier == "regle"
    assert decision.tool_call.name == "lancer_de"
    assert decision.tool_call.arguments == {"faces": 20}
    assert llm.calls == []            # aucune inférence


def test_un_skill_sans_argument_se_declenche_directement() -> None:
    llm = MockLLM()
    decision = Router(llm).route("quelle heure est-il", CATALOGUE)
    assert decision.tier == "regle"
    assert decision.tool_call.name == "heure"
    assert decision.tool_call.arguments == {}
    assert llm.calls == []


def test_sans_nombre_les_valeurs_par_defaut_sont_gardees() -> None:
    decision = Router(MockLLM()).route("lance un dé", CATALOGUE)
    assert decision.tool_call.arguments == {}


def test_l_ambiguite_numerique_est_renvoyee_au_modele() -> None:
    # Deux paramètres numériques et deux nombres : l'étage déterministe
    # refuse de deviner qui va où.
    llm = MockLLM([json.dumps({"outil": "minuteur"}), json.dumps({"minutes": 5, "secondes": 30})])
    decision = Router(llm).route("mets un minuteur de 5 minutes et 30 secondes", CATALOGUE)
    assert decision.tier == "llm"
    assert decision.tool_call.arguments == {"minutes": 5, "secondes": 30}


def test_un_argument_obligatoire_passe_par_le_modele() -> None:
    llm = MockLLM([json.dumps({"outil": "repete"}), json.dumps({"texte": "bonjour"})])
    decision = Router(llm).route("répète après moi bonjour", CATALOGUE)
    assert decision.tier == "llm"
    assert decision.tool_call.name == "repete"
    assert decision.tool_call.arguments == {"texte": "bonjour"}


def test_le_choix_de_l_outil_est_contraint_par_une_enumeration() -> None:
    llm = MockLLM([json.dumps({"outil": "outil_invente"})])
    decision = Router(llm).route("fais quelque chose d'inconnu", CATALOGUE)

    # Un nom halluciné est ramené à « aucun » plutôt que propagé.
    assert decision.tool_call is None
    assert decision.tier == "conversation"
    # Et le schéma envoyé au moteur énumère bien les noms possibles.
    enumeration = llm.calls[0]["json_schema"]["properties"]["outil"]["enum"]
    assert NO_TOOL in enumeration
    assert set(enumeration) - {NO_TOOL} <= set(CATALOGUE)


def test_aucun_outil_bascule_en_conversation() -> None:
    llm = MockLLM([json.dumps({"outil": NO_TOOL}), "Il fait beau."])
    routeur = Router(llm)
    decision = routeur.route("raconte-moi une histoire", CATALOGUE)
    assert decision.tier == "conversation"
    assert routeur.answer_freely("raconte-moi une histoire", [], "persona") == "Il fait beau."


def test_les_arguments_sont_contraints_par_le_schema_de_l_outil() -> None:
    llm = MockLLM([json.dumps({"outil": "lancer_de"}), json.dumps({"faces": 12})])
    Router(llm, direct_threshold=2.0).route("un truc avec des faces", CATALOGUE)

    schema_arguments = llm.calls[1]["json_schema"]
    assert schema_arguments["properties"]["faces"]["type"] == "integer"
    assert schema_arguments["additionalProperties"] is False


def test_la_liste_courte_est_bornee() -> None:
    llm = MockLLM([json.dumps({"outil": NO_TOOL})])
    Router(llm, direct_threshold=2.0, shortlist_size=2).route("peu importe", CATALOGUE)
    enumeration = llm.calls[0]["json_schema"]["properties"]["outil"]["enum"]
    assert len(enumeration) == 3   # 2 outils + « aucun »


def test_une_reponse_non_json_ne_casse_pas_le_routage() -> None:
    # Filet pour un moteur qui n'appliquerait pas la contrainte de décodage.
    llm = MockLLM(["je ne sais pas répondre en JSON"])
    decision = Router(llm, direct_threshold=2.0).route("peu importe", CATALOGUE)
    assert decision.tool_call is None


def test_le_json_noye_dans_du_bavardage_est_recupere() -> None:
    llm = MockLLM(['Voici : {"outil": "heure"} voilà.', "{}"])
    decision = Router(llm, direct_threshold=2.0).route("peu importe", CATALOGUE)
    assert decision.tool_call.name == "heure"


def test_un_skill_en_quarantaine_n_est_plus_propose() -> None:
    DES.quarantined = True
    try:
        decision = Router(MockLLM([json.dumps({"outil": NO_TOOL})])).route(
            "lance un dé à vingt faces", CATALOGUE
        )
        assert decision.tool_call is None or decision.tool_call.name != "lancer_de"
    finally:
        DES.quarantined = False


def test_sans_aucune_competence_on_repond_sans_planter() -> None:
    decision = Router(MockLLM()).route("bonjour", {})
    assert decision.tool_call is None
    assert decision.tier == "conversation"


# --- l'étage déterministe, préparé et élagué ---------------------------------

def _similarity_naive(gauche: str, droite: str) -> float:
    """La formule d'origine, gardée ici comme témoin.

    C'est elle qui définit ce que « le même score » veut dire : la version en
    production prépare les phrases et élague avec une borne, deux changements
    qui doivent être strictement sans effet sur le résultat.
    """
    from difflib import SequenceMatcher

    from lily.core.text import normalize, tokenize

    jetons_g, jetons_d = set(tokenize(gauche)), set(tokenize(droite))
    if not jetons_g or not jetons_d:
        return 0.0
    partages = len(jetons_g & jetons_d)
    if partages:
        precision, rappel = partages / len(jetons_d), partages / len(jetons_g)
        f1 = 2 * precision * rappel / (precision + rappel)
    else:
        f1 = 0.0
    return 0.7 * f1 + 0.3 * SequenceMatcher(None, normalize(gauche), normalize(droite)).ratio()


@pytest.mark.parametrize(
    ("gauche", "droite"),
    [
        ("mets un minuteur de dix minutes", "minuteur de trois minutes pour les pâtes"),
        ("quelle heure est-il", "donne-moi l'heure"),
        ("relance-moi le bazar", "lance le pipeline du projet"),
        ("azertyuiop", "quelle heure est-il"),
        ("", "quelque chose"),
        ("identique", "identique"),
        ("accentué déjà là", "accentue deja la"),
    ],
)
def test_la_preparation_ne_change_aucun_score(gauche: str, droite: str) -> None:
    assert similarity(gauche, droite) == pytest.approx(_similarity_naive(gauche, droite))


def test_l_elagage_ne_change_aucun_score() -> None:
    """`real_quick_ratio()` majore toujours `ratio()` : quand le plafond ne bat
    pas un score déjà obtenu, ne pas calculer le score exact ne peut rien
    changer. On le vérifie plutôt que de le croire."""
    phrases = ["mets un minuteur", "quelle heure est-il", "note ça", "lance le pipeline"]
    for gauche, droite in itertools.product(phrases, repeat=2):
        g, d = PhrasePreparee.de(gauche), PhrasePreparee.de(droite)
        exact = similarity_preparee(g, d)
        for plancher in (0.0, 0.1, 0.5, exact - 1e-9, exact, 1.0):
            elague = similarity_preparee(g, d, plancher)
            assert elague == exact or elague == 0.0
            if elague == 0.0:
                assert exact <= plancher + 1e-9, "un score qui pouvait gagner a été élagué"


def test_le_score_des_competences_est_celui_de_la_formule_temoin() -> None:
    """Le chemin complet : préparation, cache par compétence, élagage — contre
    la formule naïve appliquée phrase par phrase."""
    routeur = Router(MockLLM())
    for phrase in ("lance un dé à vingt faces", "quelle heure est-il",
                   "quelque chose de totalement hors sujet", ""):
        obtenus = {c.name: c.score for c in routeur.score_skills(phrase, CATALOGUE)}
        for nom, spec in CATALOGUE.items():
            attendu = max(
                [_similarity_naive(phrase, e) * 1.0 for e in spec.examples]
                + [_similarity_naive(phrase, nom.replace("_", " ")) * 0.85]
                + ([_similarity_naive(phrase, spec.description) * 0.45] if spec.description else []),
                default=0.0,
            )
            assert obtenus[nom] == pytest.approx(round(attendu, 3))

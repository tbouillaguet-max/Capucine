"""L'index sémantique : découper, vectoriser, retrouver — et se rabattre."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from lily.core.config import Config
from lily.core.conversation import Conversation
from lily.core.engines.embeddings.hachage import HachageEmbeddings
from lily.core.engines.embeddings.ollama import OllamaEmbeddings
from lily.core.engines.factory import build_embeddings
from lily.core.engines.llm.mock import MockLLM
from lily.core.errors import EngineUnavailable
from lily.core.interfaces.embeddings import EmbeddingEngine
from lily.core.pipeline import Pipeline
from lily.core.registry import PluginRegistry
from lily.core.router import Router
from lily.core.semantique import (
    Connaissances,
    IndexPlein,
    decouper,
    depuis_config,
)

RAPPORT = (
    "Le résultat net du premier trimestre s'établit à une perte de douze mille "
    "quatre cents euros, contre un bénéfice au trimestre précédent."
    "\n\n"
    "Le backtest du portefeuille d'options a tourné en quatre heures sur le GPU."
)


class VectoriseurEcrit(EmbeddingEngine):
    """Vectoriseur piloté par le test : chaque texte a sa coordonnée.

    Un vrai modèle rendrait des nombres qu'on ne peut pas prédire ; ici on
    écrit la géométrie qu'on veut éprouver.
    """

    name = "ecrit"
    model = "ecrit-3"

    def __init__(self, table: dict[str, list[float]] | None = None) -> None:
        self.table = table or {}
        self.appels: list[list[str]] = []

    def available(self) -> bool:
        return True

    def encode(self, textes):
        self.appels.append(list(textes))
        return [self.table.get(texte.strip(), [0.0, 0.0, 1.0]) for texte in textes]


@pytest.fixture
def index(tmp_path: Path):
    instance = Connaissances(tmp_path / "index.sqlite", HachageEmbeddings())
    yield instance
    instance.fermer()


@pytest.fixture
def index_lexical(tmp_path: Path):
    instance = Connaissances(tmp_path / "lexical.sqlite", None)
    yield instance
    instance.fermer()


# --- découpage ---------------------------------------------------------------

def test_un_texte_court_reste_entier() -> None:
    assert decouper("Trois mots seulement.") == ["Trois mots seulement."]


def test_le_decoupage_suit_les_paragraphes() -> None:
    fragments = decouper("a" * 500 + "\n\n" + "b" * 500, taille=600, recouvrement=0)
    assert len(fragments) == 2
    assert fragments[0].startswith("a") and fragments[1].startswith("b")


def test_une_phrase_plus_longue_que_le_fragment_est_coupee() -> None:
    fragments = decouper("x" * 2500, taille=900, recouvrement=100)
    assert len(fragments) >= 3
    assert all(len(fragment) <= 900 for fragment in fragments)


def test_le_recouvrement_evite_de_perdre_une_frontiere() -> None:
    # Une réponse à cheval sur deux fragments doit survivre dans au moins un.
    texte = "\n\n".join(f"Paragraphe {numero} " + "mot " * 60 for numero in range(6))
    fragments = decouper(texte, taille=500, recouvrement=120)
    assert len(fragments) > 1
    assert any(fragments[0][-40:] in fragment for fragment in fragments[1:])


def test_un_texte_vide_ne_produit_rien() -> None:
    assert decouper("   \n  ") == []


# --- indexation --------------------------------------------------------------

def test_indexer_puis_retrouver(index: Connaissances) -> None:
    assert index.indexer("rapport.md", RAPPORT) == 1
    (passage,) = index.chercher("perte du premier trimestre", limite=1)
    assert "perte de douze mille" in passage.texte
    assert passage.reference == "rapport.md"


def test_reindexer_remplace_au_lieu_d_empiler(index: Connaissances) -> None:
    index.indexer("rapport.md", RAPPORT)
    index.indexer("rapport.md", "Un contenu entièrement différent, plus court.")
    assert index.statistiques()["fragments"] == 1
    # L'ancien contenu a disparu. On regarde le texte rendu plutôt que le
    # nombre de résultats : le vectoriseur de repli est trop grossier pour
    # qu'une absence de réponse prouve quoi que ce soit.
    trouves = [passage.texte for passage in index.chercher("douze mille", limite=5)]
    assert all("douze mille" not in texte for texte in trouves)


def test_un_document_inchange_est_reconnu(index: Connaissances) -> None:
    index.indexer("rapport.md", RAPPORT, empreinte="abc123")
    assert index.deja_indexe("rapport.md", "abc123")
    assert not index.deja_indexe("rapport.md", "autre")
    assert not index.deja_indexe("inconnu.md", "abc123")


def test_sans_empreinte_on_ne_saute_jamais(index: Connaissances) -> None:
    index.indexer("rapport.md", RAPPORT)
    assert not index.deja_indexe("rapport.md", "")


def test_changer_de_modele_invalide_l_index(tmp_path: Path) -> None:
    fichier = tmp_path / "index.sqlite"
    premier = Connaissances(fichier, HachageEmbeddings())
    premier.indexer("rapport.md", RAPPORT, empreinte="abc")
    assert premier.deja_indexe("rapport.md", "abc")
    premier.fermer()

    # Deux modèles produisent deux espaces vectoriels incomparables : ce qui a
    # été indexé avec l'un ne doit pas être interrogé avec l'autre.
    second = Connaissances(fichier, VectoriseurEcrit())
    try:
        assert not second.deja_indexe("rapport.md", "abc")
    finally:
        second.fermer()


def test_l_index_dit_qu_il_est_plein(tmp_path: Path) -> None:
    petit = Connaissances(tmp_path / "petit.sqlite", HachageEmbeddings(), fragments_max=2)
    try:
        with pytest.raises(IndexPlein, match="limite"):
            petit.indexer("gros.md", "\n\n".join("phrase " * 200 for _ in range(6)))
    finally:
        petit.fermer()


def test_oublier_un_document(index: Connaissances) -> None:
    index.indexer("rapport.md", RAPPORT)
    index.indexer("notes.md", "Une note sans rapport avec le reste.")
    assert index.oublier("rapport.md") == 1
    assert [reference for reference, _ in index.references()] == ["notes.md"]


# --- la géométrie de la recherche -------------------------------------------

def test_le_plus_proche_sort_en_premier(tmp_path: Path) -> None:
    moteur = VectoriseurEcrit({
        "Le chat dort.": [1.0, 0.0, 0.0],
        "Le chien dort.": [0.9, 0.44, 0.0],
        "Le cours du pétrole monte.": [0.0, 0.0, 1.0],
        "question": [1.0, 0.0, 0.0],
    })
    index = Connaissances(tmp_path / "i.sqlite", moteur)
    try:
        for numero, texte in enumerate(moteur.table):
            if texte != "question":
                index.indexer(f"doc{numero}", texte)
        passages = index.chercher("question", limite=3)
        assert [passage.texte for passage in passages] == ["Le chat dort.", "Le chien dort."]
        # Le troisième est orthogonal : hors sujet, donc écarté plutôt que rendu.
        assert passages[0].score > passages[1].score
    finally:
        index.fermer()


def test_les_vecteurs_sont_normalises_a_l_indexation(tmp_path: Path) -> None:
    # Un moteur qui ne normalise pas doit quand même donner une similarité
    # cosinus correcte : c'est l'index qui s'en charge.
    moteur = VectoriseurEcrit({"court": [3.0, 0.0, 0.0], "long": [100.0, 0.0, 0.0]})
    index = Connaissances(tmp_path / "n.sqlite", moteur)
    try:
        index.indexer("a", "court")
        index.indexer("b", "long")
        scores = [passage.score for passage in index.chercher("court", limite=2)]
        assert scores == pytest.approx([1.0, 1.0], abs=1e-4)
    finally:
        index.fermer()


def test_la_recherche_peut_se_limiter_a_une_source(index: Connaissances) -> None:
    index.indexer("rapport.md", RAPPORT, source="document")
    index.indexer("conversation 1", "Nous parlions du backtest hier.", source="conversation")
    documents = index.chercher("backtest", limite=5, sources=("document",))
    assert {passage.source for passage in documents} == {"document"}


def test_un_passage_dit_d_ou_il_vient(index: Connaissances) -> None:
    index.indexer("/home/moi/rapport.md", RAPPORT, ancres=["page 1"])
    (passage,) = index.chercher("perte", limite=1)
    assert passage.provenance == "rapport.md — page 1"
    assert passage.citer().startswith("[rapport.md — page 1]")


# --- le repli lexical --------------------------------------------------------

def test_sans_vectoriseur_la_recherche_marche_quand_meme(index_lexical) -> None:
    assert not index_lexical.vectoriel
    index_lexical.indexer("rapport.md", RAPPORT)
    (passage,) = index_lexical.chercher("backtest", limite=1)
    assert "backtest" in passage.texte


def test_le_repli_lexical_ne_confond_pas_les_documents(index_lexical) -> None:
    index_lexical.indexer("a.md", "Il est question de volatilité implicite.")
    index_lexical.indexer("b.md", "Il est question de la cantine du mardi.")
    (passage,) = index_lexical.chercher("volatilité", limite=1)
    assert passage.reference == "a.md"


def test_un_vectoriseur_qui_tombe_en_panne_bascule_en_lexical(tmp_path: Path) -> None:
    class MoteurCasse(VectoriseurEcrit):
        def encode(self, textes):
            raise EngineUnavailable("le service a disparu")

    index = Connaissances(tmp_path / "p.sqlite", HachageEmbeddings())
    try:
        index.indexer("rapport.md", RAPPORT)
        index.moteur = MoteurCasse()
        # La question ne peut plus être vectorisée : plutôt que de ne rien
        # rendre, on retombe sur le plein texte.
        (passage,) = index.chercher("backtest", limite=1)
        assert "backtest" in passage.texte
    finally:
        index.fermer()


# --- l'indexation des conversations, en fond ---------------------------------

def _attendre(condition, delai: float = 5.0) -> bool:
    limite = time.monotonic() + delai
    while time.monotonic() < limite:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_un_tour_de_conversation_rejoint_l_index(index: Connaissances) -> None:
    index.indexer_le_tour(
        "conversation 1",
        "où en est le backtest du portefeuille d'options",
        "Il a tourné en quatre heures sur le GPU.",
    )
    assert _attendre(lambda: index.statistiques()["tours"] == 1)
    (passage,) = index.chercher("backtest", limite=1, sources=("conversation",))
    assert "quatre heures" in passage.texte


def test_un_tour_trop_court_n_est_pas_indexe(index: Connaissances) -> None:
    index.indexer_le_tour("conversation 1", "l'heure ?", "midi")
    assert not _attendre(lambda: index.statistiques()["tours"] > 0, delai=0.4)


def test_l_indexation_des_conversations_se_coupe(tmp_path: Path) -> None:
    index = Connaissances(
        tmp_path / "c.sqlite", HachageEmbeddings(), indexer_les_conversations=False
    )
    try:
        index.indexer_le_tour("conversation 1", "une question bien assez longue pour compter", "une réponse")
        assert not _attendre(lambda: index.statistiques()["tours"] > 0, delai=0.4)
    finally:
        index.fermer()


# --- le pipeline nourrit l'index --------------------------------------------

PLUGIN = '''
from lily.plugin import skill

@skill(description="Donne l'heure.", examples=["quelle heure est-il"])
def heure() -> str:
    return "Il est midi, et le backtest du portefeuille tourne encore."
'''


def test_le_pipeline_confie_ses_tours_a_l_index(
    ecrire_plugin, dossier_plugins, index
) -> None:
    ecrire_plugin("essai.py", PLUGIN)
    registry = PluginRegistry([dossier_plugins], data_root=dossier_plugins.parent / "d")
    registry.load_all()
    pipeline = Pipeline(
        registry, Router(MockLLM()), Conversation(persona="p"), connaissances=index,
    )
    asyncio.run(pipeline.handle("quelle heure est-il"))
    assert _attendre(lambda: index.statistiques()["tours"] == 1)


def test_un_index_en_panne_ne_casse_pas_un_tour(ecrire_plugin, dossier_plugins) -> None:
    class IndexCasse:
        def indexer_le_tour(self, *args, **kwargs):
            raise OSError("disque plein")

    ecrire_plugin("essai.py", PLUGIN)
    registry = PluginRegistry([dossier_plugins], data_root=dossier_plugins.parent / "d")
    registry.load_all()
    pipeline = Pipeline(
        registry, Router(MockLLM()), Conversation(persona="p"), connaissances=IndexCasse(),
    )
    resultat = asyncio.run(pipeline.handle("quelle heure est-il"))
    assert resultat.speak.startswith("Il est midi")


# --- le contexte donné au modèle ---------------------------------------------

def test_le_contexte_porte_les_provenances(index: Connaissances) -> None:
    index.indexer("/tmp/rapport.md", RAPPORT)
    contexte = index.contexte("perte du premier trimestre")
    assert "[rapport.md]" in contexte
    assert "douze mille" in contexte


def test_le_contexte_respecte_sa_limite(index: Connaissances) -> None:
    for numero in range(5):
        index.indexer(f"doc{numero}.md", f"Le backtest numéro {numero} " + "détail " * 60)
    contexte = index.contexte("backtest", limite=5, caracteres_max=500)
    assert len(contexte) <= 500


# --- la fabrique et la configuration -----------------------------------------

def test_le_vectoriseur_se_desactive(tmp_path: Path) -> None:
    assert build_embeddings(Config({"connaissances": {"active": False}})) is None
    assert build_embeddings(Config({"connaissances": {"engine": "aucun"}})) is None


def test_un_vectoriseur_injoignable_ne_fait_pas_echouer_le_demarrage() -> None:
    # Le port est volontairement absurde : rien n'écoute. Le démarrage doit
    # continuer, la recherche restera lexicale.
    config = Config({"connaissances": {"engine": "ollama", "host": "http://127.0.0.1:1"}})
    assert build_embeddings(config) is None


def test_le_vectoriseur_refuse_un_hote_distant() -> None:
    with pytest.raises(EngineUnavailable, match="non local"):
        OllamaEmbeddings(host="http://192.168.1.50:11434")


def test_la_fabrique_avale_le_refus_d_un_hote_distant() -> None:
    # Une configuration fautive ne doit pas empêcher Lily de démarrer.
    config = Config({"connaissances": {"engine": "ollama", "host": "http://exemple.com"}})
    assert build_embeddings(config) is None


def test_l_index_se_desactive(tmp_path: Path) -> None:
    assert depuis_config(Config({"connaissances": {"active": False}})) is None


def test_l_index_suit_le_fichier_de_la_memoire(tmp_path: Path) -> None:
    config = Config({"memoire": {"fichier": str(tmp_path / "memoire.sqlite")}})
    index = depuis_config(config)
    try:
        assert index.chemin == tmp_path / "memoire.sqlite"
    finally:
        index.fermer()


def test_le_moteur_par_hachage_est_deterministe() -> None:
    moteur = HachageEmbeddings()
    (premier,) = moteur.encode(["le backtest du portefeuille"])
    (second,) = moteur.encode(["le backtest du portefeuille"])
    assert premier == second
    assert sum(valeur * valeur for valeur in premier) == pytest.approx(1.0)


def test_le_moteur_ollama_vectorise_par_lot(monkeypatch) -> None:
    """La signature réelle du client est vérifiée ici, pas de mémoire.

    ``Client.embed(model=…, input=[…])`` rend un objet dont ``.embeddings``
    porte un vecteur par texte, dans l'ordre.
    """
    class FausseReponse:
        embeddings = [[1.0, 0.0], [0.0, 1.0]]

    class FauxClient:
        def __init__(self) -> None:
            self.recu: dict = {}

        def embed(self, **kwargs):
            self.recu = kwargs
            return FausseReponse()

    moteur = OllamaEmbeddings()
    client = FauxClient()
    monkeypatch.setattr(moteur, "_get_client", lambda: client)
    assert moteur.encode(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert client.recu["model"] == "nomic-embed-text"
    assert client.recu["input"] == ["a", "b"]


def test_le_moteur_ollama_exige_le_modele_tire(monkeypatch) -> None:
    class Entree:
        def __init__(self, nom: str) -> None:
            self.model = nom

    class Listing:
        def __init__(self, noms) -> None:
            self.models = [Entree(nom) for nom in noms]

    moteur = OllamaEmbeddings()
    monkeypatch.setattr(moteur, "_get_client", lambda: type(
        "C", (), {"list": staticmethod(lambda: Listing(["qwen2.5:7b-instruct-q4_K_M"]))}
    )())
    assert not moteur.available()      # le service répond, le modèle manque

    monkeypatch.setattr(moteur, "_get_client", lambda: type(
        "C", (), {"list": staticmethod(lambda: Listing(["nomic-embed-text:latest"]))}
    )())
    assert moteur.available()          # « :latest » est le même modèle


# --- les compétences livrées -------------------------------------------------

def test_les_competences_de_connaissances_repondent(index, tmp_path) -> None:
    from lily.core import plugin as contrat
    from lily.core.config import PROJECT_ROOT

    contrat.set_connaissances(index)
    contrat.set_model_access(lambda prompt, **kwargs: "D'après le rapport, la perte est de douze mille euros.")
    registry = PluginRegistry([PROJECT_ROOT / "plugins"], data_root=tmp_path / "d")
    registry.load_all()
    try:
        assert registry.call("mes_connaissances").speak.startswith("Je n'ai encore rien indexé")

        index.indexer("/tmp/rapport.md", RAPPORT, source="document", empreinte="x")
        resultat = registry.call("que_sais_tu_sur", {"sujet": "la perte du premier trimestre"})
        assert resultat.ok
        assert "douze mille" in resultat.speak
        assert "rapport.md" in resultat.display      # la provenance est montrée

        assert registry.call("passages_sur", {"sujet": "backtest"}).ok
        assert "1 document" in registry.call("mes_connaissances").display

        # Oublier détruit du travail d'indexation : Lily demande d'abord.
        demande = registry.call("oublier_ce_que_tu_as_lu", {"document": "rapport.md"})
        assert demande.needs_confirmation
        assert index.statistiques()["fragments"] == 1

        oubli = registry.call(
            "oublier_ce_que_tu_as_lu", {"document": "rapport.md"}, confirmed=True
        )
        assert oubli.ok and "1 document" in oubli.speak
        assert index.statistiques()["fragments"] == 0
    finally:
        for nom in list(registry.plugins):
            registry.unload(nom, notify=False)
        contrat.set_connaissances(None)
        contrat.set_model_access(None)


def test_sans_index_la_competence_refuse_proprement(tmp_path) -> None:
    from lily.core import plugin as contrat
    from lily.core.config import PROJECT_ROOT

    contrat.set_connaissances(None)
    registry = PluginRegistry([PROJECT_ROOT / "plugins"], data_root=tmp_path / "d")
    registry.load_all()
    try:
        resultat = registry.call("mes_connaissances")
        assert not resultat.ok
        assert "désactivé" in resultat.speak
    finally:
        for nom in list(registry.plugins):
            registry.unload(nom, notify=False)


# --- garde-fous du découpage ------------------------------------------------

def test_un_recouvrement_aussi_large_que_la_taille_ne_boucle_pas() -> None:
    """Les deux valeurs viennent de la configuration, et rien ne les empêchait
    d'être égales — auquel cas `phrase[taille - recouvrement:]` valait
    `phrase[0:]`, donc une boucle sans fin au premier document indexé."""
    fragments = decouper("mot " * 400, taille=100, recouvrement=100)
    assert fragments
    assert all(len(fragment) <= 101 for fragment in fragments)


def test_un_recouvrement_absurde_est_borne_sans_rien_perdre() -> None:
    fragments = decouper("phrase sans ponctuation " * 60, taille=120, recouvrement=5000)
    assert fragments
    assert "".join(fragments).replace("\n", "").strip()

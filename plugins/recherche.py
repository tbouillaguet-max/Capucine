"""Recherche sur le web, et lecture d'une page.

**Ce plugin sort de la machine.** C'est la seule entorse à la règle numéro un
du projet — « tout tourne en local » — et elle est délibérée : elle vit dans un
plugin, pas dans le cœur. Retirez le fichier et Lily redevient
intégralement hors-ligne. Elle annonce d'ailleurs qu'elle va sur le réseau
plutôt que de le faire en silence.

Trois moteurs, par ordre de préférence :

* ``searxng`` — une instance que **vous** hébergez. C'est le choix cohérent
  avec l'esprit du projet : le méta-moteur interroge Google et les autres,
  mais depuis votre machine, sans compte ni clé, sans profilage.
* ``google`` — l'API officielle *Custom Search JSON*. Demande une clé et un
  identifiant de moteur, 100 requêtes par jour gratuites.
* ``duckduckgo`` — sans clé, en lisant la page HTML. Sans garantie : c'est du
  décorticage de page, ça casse le jour où la mise en page change.

Aucune dépendance ajoutée : ``urllib`` et ``html.parser`` suffisent.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from lily.plugin import get_config, get_logger, skill

CONFIG_DEFAULTS = {
    "moteur": "searxng",                    # searxng | google | duckduckgo
    "searxng_url": "http://127.0.0.1:8888",
    "google_cle_api": "",
    "google_cx": "",
    "resultats": 3,
    "delai_s": 10.0,
    "taille_page_max_ko": 400,
    "agent": "Lily/1.0 (assistante vocale locale)",
}


class _Texte(HTMLParser):
    """Extrait le texte d'une page. Volontairement rustique : on veut le sens
    général pour le lire à voix haute, pas une reconstruction fidèle."""

    IGNORES = {"script", "style", "noscript", "svg", "head", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.morceaux: list[str] = []
        self._profondeur_ignoree = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.IGNORES:
            self._profondeur_ignoree += 1

    def handle_endtag(self, tag):
        if tag in self.IGNORES and self._profondeur_ignoree:
            self._profondeur_ignoree -= 1
        elif tag in ("p", "div", "li", "h1", "h2", "h3", "br", "tr"):
            self.morceaux.append("\n")

    def handle_data(self, data):
        if not self._profondeur_ignoree and data.strip():
            self.morceaux.append(data.strip())

    def texte(self) -> str:
        brut = " ".join(self.morceaux)
        lignes = [ligne.strip() for ligne in brut.split("\n")]
        return "\n".join(ligne for ligne in lignes if ligne)


class _ResultatsDuckDuckGo(HTMLParser):
    """Décortique la page HTML de DuckDuckGo. Fragile par nature."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.resultats: list[dict] = []
        self._dans_titre = False
        self._dans_extrait = False
        self._url = ""

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attributs = dict(attrs)
        classes = attributs.get("class", "")
        if "result__a" in classes:
            self._dans_titre = True
            self._url = _url_reelle(attributs.get("href", ""))
            self.resultats.append({"titre": "", "url": self._url, "extrait": ""})
        elif "result__snippet" in classes and self.resultats:
            self._dans_extrait = True

    def handle_endtag(self, tag):
        if tag == "a":
            self._dans_titre = self._dans_extrait = False

    def handle_data(self, data):
        if not self.resultats:
            return
        if self._dans_titre:
            self.resultats[-1]["titre"] += data.strip()
        elif self._dans_extrait:
            self.resultats[-1]["extrait"] += data.strip()


def _url_reelle(href: str) -> str:
    """DuckDuckGo enrobe ses liens dans une redirection ; on la déplie."""
    if "uddg=" in href:
        parametres = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        return parametres.get("uddg", [href])[0]
    return href


def _telecharger(url: str, delai: float, taille_max_ko: int = 400) -> str:
    requete = urllib.request.Request(
        url, headers={"User-Agent": str(get_config("agent", "Lily/1.0"))}
    )
    with urllib.request.urlopen(requete, timeout=delai) as reponse:  # noqa: S310
        brut = reponse.read(taille_max_ko * 1024)
    return brut.decode("utf-8", errors="replace")


# --- moteurs ----------------------------------------------------------------

def _searxng(question: str, nombre: int, delai: float) -> list[dict]:
    base = str(get_config("searxng_url", "http://127.0.0.1:8888")).rstrip("/")
    url = f"{base}/search?" + urllib.parse.urlencode(
        {"q": question, "format": "json", "language": "fr"}
    )
    donnees = json.loads(_telecharger(url, delai))
    return [
        {"titre": r.get("title", ""), "url": r.get("url", ""), "extrait": r.get("content", "")}
        for r in donnees.get("results", [])[:nombre]
    ]


def _google(question: str, nombre: int, delai: float) -> list[dict]:
    cle, cx = str(get_config("google_cle_api", "")), str(get_config("google_cx", ""))
    if not cle or not cx:
        raise RuntimeError(
            "La recherche Google demande une clé d'API et un identifiant de moteur. "
            "Renseignez plugins.recherche.google_cle_api et google_cx dans la "
            "configuration, ou passez à moteur = \"searxng\"."
        )
    url = "https://www.googleapis.com/customsearch/v1?" + urllib.parse.urlencode(
        {"key": cle, "cx": cx, "q": question, "num": min(nombre, 10), "hl": "fr"}
    )
    donnees = json.loads(_telecharger(url, delai))
    return [
        {"titre": r.get("title", ""), "url": r.get("link", ""), "extrait": r.get("snippet", "")}
        for r in donnees.get("items", [])[:nombre]
    ]


def _duckduckgo(question: str, nombre: int, delai: float) -> list[dict]:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": question})
    analyseur = _ResultatsDuckDuckGo()
    analyseur.feed(_telecharger(url, delai))
    return analyseur.resultats[:nombre]


MOTEURS = {"searxng": _searxng, "google": _google, "duckduckgo": _duckduckgo}


# --- compétences ------------------------------------------------------------

@skill(
    description="Cherche une information sur le web et résume les résultats.",
    examples=[
        "cherche sur internet la météo à Amiens",
        "fais une recherche sur les taux directeurs de la BCE",
        "qu'est-ce que dit le web sur la volatilité implicite",
    ],
    timeout=30.0,
)
def chercher(question: str, nombre: int = 0) -> dict:
    """Interroge le moteur de recherche configuré.

    Args:
        question: Ce qu'il faut chercher.
        nombre: Combien de résultats. Zéro prend la valeur configurée.
    """
    question = question.strip()
    if not question:
        return {"speak": "Que dois-je chercher ?", "display": "requête vide"}

    nom = str(get_config("moteur", "searxng"))
    moteur = MOTEURS.get(nom)
    if moteur is None:
        return {
            "speak": "Le moteur de recherche configuré ne m'est pas connu.",
            "display": f"moteur inconnu : {nom} (choix : {', '.join(MOTEURS)})",
        }

    nombre = int(nombre) or int(get_config("resultats", 3))
    delai = float(get_config("delai_s", 10.0))
    get_logger().info("Recherche « %s » via %s — sortie réseau.", question, nom)

    try:
        resultats = moteur(question, nombre, delai)
    except urllib.error.URLError as exc:
        return {
            "speak": "Je n'arrive pas à joindre le moteur de recherche.",
            "display": f"{nom} injoignable : {exc.reason}",
        }
    except RuntimeError as exc:
        return {"speak": str(exc), "display": str(exc)}
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "speak": "Le moteur de recherche a répondu quelque chose que je ne comprends pas.",
            "display": f"réponse illisible de {nom} : {exc}",
        }

    if not resultats:
        return {"speak": f"Je ne trouve rien sur « {question} ».", "display": "0 résultat"}

    parle = " ".join(
        f"{r['titre'].rstrip('.')}. {r['extrait'][:200]}" for r in resultats[:2] if r["titre"]
    )
    return {
        "speak": parle or "J'ai des résultats, mais aucun résumé lisible.",
        "display": "\n".join(f"- {r['titre']}\n  {r['url']}\n  {r['extrait'][:160]}" for r in resultats),
    }


@skill(
    description="Lit une page web et en donne le contenu texte.",
    examples=["lis cette page", "résume la page", "va voir ce lien"],
    timeout=30.0,
)
def lire_page(url: str) -> dict:
    """Télécharge une page et en extrait le texte.

    Args:
        url: L'adresse complète de la page.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    delai = float(get_config("delai_s", 10.0))
    taille = int(get_config("taille_page_max_ko", 400))
    get_logger().info("Lecture de %s — sortie réseau.", url)

    try:
        html = _telecharger(url, delai, taille)
    except urllib.error.URLError as exc:
        return {"speak": "Je n'arrive pas à ouvrir cette page.", "display": f"{url} : {exc.reason}"}

    analyseur = _Texte()
    analyseur.feed(html)
    texte = analyseur.texte()
    if not texte:
        return {"speak": "Cette page ne contient pas de texte lisible.", "display": url}
    return {"speak": texte[:600], "display": texte[:4000]}

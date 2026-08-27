# Capucine

Assistante vocale francophone, **entièrement locale**. Aucun service tiers,
aucune clé d'API, aucun appel réseau sortant : elle fonctionne le Wi-Fi coupé.

La contrainte qui prime sur toutes les autres décisions d'architecture :
**ajouter une capacité, c'est déposer un fichier Python dans `plugins/`.** Sans
redémarrer l'assistante, et sans jamais toucher au cœur.

> **État : étape 1 sur 5 terminée.** La boucle « phrase → choix d'outil →
> plugin → réponse » fonctionne en mode texte. L'audio arrive aux étapes 2 et 3.

---

## Installation

Python 3.11 ou plus récent. Le cœur ne dépend que de la bibliothèque standard ;
tout le reste est optionnel et chargé paresseusement.

```bash
git clone https://github.com/tbouillaguet-max/Capucine.git
cd Capucine
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
```

Essayez immédiatement, sans aucun modèle :

```bash
python main.py --text --llm mock
```

### Ajouter un modèle de langage local

**Ollama (recommandé sur PC)** — [ollama.com/download](https://ollama.com/download),
puis :

```bash
pip install -e ".[llm-ollama]"
ollama pull qwen2.5:7b-instruct-q4_K_M     # PC
ollama pull qwen2.5:3b-instruct-q4_K_M     # Raspberry Pi
python main.py --text
```

Ollama tourne comme un service **sur votre machine**. Capucine valide l'hôte et
refuse de démarrer s'il n'est pas une adresse de bouclage : rien ne sort de la
machine.

**llama.cpp en processus** — si vous préférez éviter le démon :

```bash
pip install -e ".[llm-llamacpp]"
# déposez un GGUF dans models/, puis dans config/pc.toml :
#   [llm]
#   engine = "llamacpp"
#   model_path = "models/qwen2.5-7b-instruct-q4_k_m.gguf"
#   n_gpu_layers = 35
```

---

## Utilisation

```bash
python main.py --text                       # boucle clavier
python main.py --text --llm mock            # sans modèle de langage
python main.py --text --once "12 * 8"       # une phrase, puis on quitte
python main.py --text --profile pi          # forcer un profil
python main.py --text --json-logs           # journal en JSON, une ligne par événement
```

Dans la boucle : `/aide` `/competences` `/plugins` `/recharge` `/oublie` `/quitter`.

---

## Écrire un plugin

Un fichier `.py` dans `plugins/`. C'est tout : pas d'enregistrement, pas
d'import à ajouter, pas de manifeste.

```python
from capucine.plugin import skill

@skill(
    description="Donne la météo actuelle ou prévue pour une ville.",
    examples=["quel temps fait-il à Amiens", "est-ce qu'il pleut demain"],
)
def meteo(ville: str, jour: str = "aujourd'hui") -> str:
    """Le corps de la docstring sert de contexte au LLM."""
    return f"Il fait 12 degrés et le ciel est couvert à {ville}."
```

Le schéma d'outil envoyé au modèle est déduit du nom, des annotations, des
valeurs par défaut et de la docstring. **Vous n'écrivez jamais de JSON.**

### Ce que le contrat vous donne

| Élément | Rôle |
|---|---|
| `@skill(description=…, examples=…)` | Déclare la compétence. `examples` n'est pas décoratif : le routeur déterministe s'en sert pour choisir l'outil **sans** solliciter le modèle. |
| `@skill(timeout=…)` | Délai maximum. Au-delà, Capucine répond qu'elle n'a pas pu exécuter la commande. |
| Docstring | Contexte pour le modèle. Les sections `Args:` ou `:param x:` documentent chaque argument. |
| `CONFIG_DEFAULTS = {…}` | Réglages du plugin, surchargés par `[plugins.<nom>]` du fichier de configuration. |
| `get_config("cle")` | Lit ces réglages. Utilisable dès le corps du module. |
| `data_dir()` | Dossier inscriptible réservé au plugin, créé à la demande. |
| `get_logger()` | Journal nommé d'après le plugin. |
| `announce("…")` | Fait parler Capucine hors d'un tour — pour une tâche de fond, un minuteur qui sonne. |
| `on_load()` / `on_unload()` | Cycle de vie : ouvrir une connexion, libérer une ressource. |

### Ce que le plugin retourne

* une **chaîne** → lue à voix haute telle quelle ;
* un **dict** `{"speak": …, "display": …}` → dissocie ce qui est dit de ce qui
  est journalisé ;
* `None` → Capucine ne dit rien.

### Règles

* **Jamais d'installation automatique de dépendance.** Un import manquant
  écarte le plugin avec un message qui nomme le paquet et la commande.
* **Un plugin ne fait jamais tomber Capucine.** Erreur à l'import, exception à
  l'exécution, dépassement de délai, `sys.exit()` : tout est confiné.
* Un plugin en échec répété est mis en quarantaine et retiré du catalogue
  proposé au modèle.

Les quatre fichiers de `plugins/` sont de la documentation vivante : lisez-les.

---

## Architecture

```
IDLE → WAKE → LISTEN → TRANSCRIBE → THINK → ACT → SPEAK → IDLE
```

Chaque étage est derrière une interface abstraite (`WakeWordEngine`,
`STTEngine`, `LLMEngine`, `TTSEngine`). Changer de moteur est une ligne de
configuration, pas une refonte.

```
capucine/
├── plugin.py               ← LE module que les plugins importent
├── app.py                  assemblage config → registre → routeur → pipeline
└── core/
    ├── pipeline.py         machine à états (asyncio)
    ├── registry.py         découverte, chargement, isolation des pannes
    ├── router.py           choix de l'outil, trois étages
    ├── schema.py           signature Python → JSON Schema
    ├── plugin.py           @skill et contrat public
    ├── config.py           TOML en couches
    ├── conversation.py     persona + mémoire courte
    ├── text.py             normalisation, nombres français
    ├── interfaces/         ABC seulement, aucune dépendance lourde
    └── engines/            implémentations, importées paresseusement
plugins/                    ← LE dossier
config/                     default.toml, pc.toml, pi.toml, persona.txt
```

### Le routage, en trois étages

Un modèle 7-8B quantifié en Q4, en français, avec une vingtaine d'outils dans
son contexte, hallucine des noms d'outils et se trompe de types d'arguments
assez souvent pour casser une démonstration. Sur Pi avec un 1-3B, c'est pire.
D'où :

1. **Étage déterministe** — score entre la phrase entendue et les `examples` du
   décorateur. Latence nulle, aucun modèle sollicité. « Lance un dé à vingt
   faces » est résolu ici, `faces=20` compris.
2. **Étage LLM, en deux passes contraintes** — d'abord le *nom* de l'outil,
   contraint par une énumération : un nom halluciné devient structurellement
   impossible. Puis les *arguments*, contraints par le schéma réel de l'outil
   choisi. Deux petites générations garanties valides valent mieux qu'une
   grande espérée valide.
3. **Étage conversationnel** — aucun outil ne convient, le modèle répond.

La sortie structurée passe par le décodage contraint (`format=<schéma>` avec
Ollama, grammaire GBNF avec llama.cpp), jamais par l'API de *function calling*
native : c'est le seul mécanisme identique sur les deux backends.

### Configuration

Quatre couches, de la plus faible à la plus forte :

1. `config/default.toml` ;
2. le profil `config/pc.toml` ou `config/pi.toml`, détecté automatiquement ;
3. les variables d'environnement `CAPUCINE_SECTION__CLE` ;
4. les options de ligne de commande.

```bash
CAPUCINE_LLM__MODEL=qwen2.5:3b-instruct-q4_K_M python main.py --text
```

---

## Tests

```bash
python -m pytest
```

87 tests, aucun modèle téléchargé, aucun périphérique audio requis.

---

## Feuille de route

| Étape | Contenu | État |
|---|---|---|
| 1 | Squelette, config, interfaces, registre de plugins, routeur, mode texte | **fait** |
| 2 | STT (`faster-whisper`) + TTS (`piper`), pipeline vocal au clavier | à venir |
| 3 | Mot d'éveil « Capucine » (`openWakeWord`), VAD (`silero`), barge-in | à venir |
| 4 | Rechargement à chaud (`watchdog`), plugins d'exemple (heure, minuteur, notes, système) | à venir |
| 5 | Profil Raspberry Pi, mesures de latence, guide d'installation par plateforme | à venir |

### Limite assumée

On ne peut pas tuer un thread en Python. Un plugin parti en boucle infinie est
*abandonné* — Capucine répond et le met en quarantaine — mais son thread
continue jusqu'à ce qu'il finisse. La seule parade réelle est le sous-processus,
qui coûte 100 à 300 ms au démarrage sur Pi et interdit l'état en mémoire (donc
le minuteur). Ce sera une option déclarée par skill, pas le défaut.

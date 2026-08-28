# Capucine

Assistante vocale francophone, **entièrement locale**. Aucun service tiers,
aucune clé d'API, aucun appel réseau sortant : elle fonctionne le Wi-Fi coupé.

La contrainte qui prime sur toutes les autres décisions d'architecture :
**ajouter une capacité, c'est déposer un fichier Python dans `plugins/`.** Sans
redémarrer l'assistante, et sans jamais toucher au cœur.

> **État : les cinq étapes sont terminées.** Le critère d'acceptation du projet
> passe : déposez un fichier dans `plugins/` pendant que Capucine tourne, elle
> annonce la nouvelle compétence et l'exécute — sans redémarrage et sans
> toucher au cœur.

---

## Le critère d'acceptation, en trois gestes

Capucine tourne. Sans l'arrêter, sans toucher à une ligne du cœur :

```bash
cat > plugins/dés.py <<'EOF'
import random
from capucine.plugin import skill

@skill(description="Lance un dé.", examples=["lance un dé", "tire un dé"])
def lancer_de(faces: int = 6) -> str:
    """Lance un dé à N faces."""
    return str(random.randint(1, faces))
EOF
```

```
Capucine › Nouvelle compétence disponible : lancer de.
Vous     › lance un dé à vingt faces
Capucine › 13
```

Aucun modèle de langage n'a été sollicité : l'étage déterministe du routeur a
reconnu l'outil sur ses `examples` et lu « vingt » comme `faces=20`. Le
scénario est rejoué en entier par `tests/test_recette.py`, avec un vrai
observateur de fichiers.

---

## Installation

Python 3.11 ou plus récent. Le cœur ne dépend que de la bibliothèque standard ;
tout le reste est optionnel et chargé paresseusement. On installe donc par
couches, et Capucine fonctionne à chaque étape.

Rien n'est jamais téléchargé automatiquement au démarrage : un assistant censé
fonctionner le Wi-Fi coupé ne sort pas sur le réseau sans qu'on le lui demande.

### Windows

```powershell
git clone https://github.com/tbouillaguet-max/Capucine.git
cd Capucine
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,reload]"

# Essayez tout de suite, sans le moindre modèle :
python main.py --text --llm mock
```

**Le modèle de langage.** Installez [Ollama](https://ollama.com/download), puis :

```powershell
pip install -e ".[llm-ollama]"
ollama pull qwen2.5:7b-instruct-q4_K_M
python main.py --text
```

**La voix.** `sounddevice` embarque PortAudio dans ses roues Windows : rien à
installer à côté.

```powershell
pip install -e ".[audio]"
python -m capucine.core.downloads tout     # voix Piper + modèle Whisper
python main.py --devices                   # repérez votre micro
python main.py --push-to-talk
```

Avec un GPU NVIDIA, `stt.device = "cuda"` dans `config/pc.toml` divise le temps
de transcription par cinq à dix. Il faut CUDA et cuDNN installés ; sans eux,
`faster-whisper` échoue au chargement — laissez `"auto"` en cas de doute.

**L'écoute permanente.**

```powershell
pip install -e ".[wake,vosk]"
pip install --no-deps silero-vad           # --no-deps n'est pas une coquille, voir plus bas
python -m capucine.core.downloads vosk
python main.py                             # dites « Capucine »
```

### Linux et Raspberry Pi

```bash
sudo apt update
sudo apt install python3 python3-venv python3-dev git libportaudio2

git clone https://github.com/tbouillaguet-max/Capucine.git
cd Capucine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,reload]"

python main.py --text --llm mock
```

**Le modèle de langage.** Ollama tourne sur ARM 64 bits :

```bash
curl -fsSL https://ollama.com/install.sh | sh
pip install -e ".[llm-ollama]"
ollama pull qwen2.5:3b-instruct-q4_K_M     # 1,5b si vous avez 2 Go de RAM
```

**La voix et l'écoute.**

```bash
pip install -e ".[audio,wake,vosk]"
pip install --no-deps silero-vad
python -m capucine.core.downloads tout --profile pi
python -m capucine.core.downloads vosk

python main.py --devices                   # repérez votre micro USB
python main.py --profile pi
```

Sur Raspberry Pi, deux réglages système comptent autant que la configuration :

```bash
# 1. De l'espace d'échange, sinon le premier chargement de Whisper échoue.
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon

# 2. Le processeur en performance : sinon il descend en fréquence pendant
#    l'inférence, exactement quand on en a besoin.
sudo apt install cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
```

Le profil `pi` est appliqué automatiquement sur une carte ARM. `python main.py
--text` affiche au démarrage les réglages qui vont décevoir sur votre machine —
Whisper trop gros pour la RAM disponible, contexte trop long, barge-in par la
voix sur haut-parleur ouvert.

### Démarrage automatique sur Raspberry Pi

```bash
sudo cp deploy/capucine.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now capucine
journalctl -u capucine -f
```

Le fichier est commenté et à adapter : utilisateur, chemins, périphérique
audio. Il journalise en JSON, ce qui permet d'agréger les latences ailleurs.

### Pourquoi `pip install --no-deps silero-vad`

Le paquet `silero-vad` importe `torch` **et** `torchaudio` dès son `__init__`,
y compris sur le chemin ONNX : plusieurs centaines de méga-octets sur un Pi,
pour un modèle de moins d'un méga-octet. Or le fichier `silero_vad.onnx` est
livré *dans* le paquet. Capucine le localise sans importer le paquet et
l'exécute avec `onnxruntime` : mêmes poids, sans la chaîne torch.

Sans lui, un VAD par énergie à plancher de bruit adaptatif prend le relais
automatiquement — moins fin dans le bruit, mais opérationnel partout.

### Le mot d'éveil demande un modèle qui n'existe pas encore

openWakeWord ne fournit aucun modèle « capucine » pré-entraîné ; il faut
l'entraîner, ce que pilote `tools/entrainer_capucine.py`. En attendant,
Capucine bascule automatiquement sur le repli Vosk à grammaire restreinte —
un décodeur qui n'a le droit de reconnaître que « capucine » et rien d'autre,
donc rapide et peu gourmand. Ce n'est pas une panne, c'est un état normal du
projet.

## Utilisation

```bash
python main.py                              # écoute permanente : dites « Capucine »
python main.py --push-to-talk               # [Entrée] pour parler, sans mot d'éveil
python main.py --no-wake                    # écoute tout, sans mot d'éveil
python main.py --barge-in eveil             # ne se tait que si l'on redit son nom
python main.py --text                       # boucle clavier
python main.py --text --llm mock            # sans modèle de langage
python main.py --text --once "12 * 8"       # une phrase, puis on quitte
python main.py --profile pi                 # forcer un profil
python main.py --json-logs                  # journal en JSON, une ligne par événement
```

Dans les deux modes : `/aide` `/competences` `/plugins` `/recharge` `/oublie`
`/quitter`. En mode vocal, une ligne vide déclenche l'écoute et tout autre
texte est traité comme si vous l'aviez dit — pratique pour éprouver la synthèse
sans parler à sa machine.

### Sans micro ni haut-parleur

Toute la chaîne vocale se pilote par fichiers, ce qui la rend testable en
intégration continue comme sur une machine sans carte son :

```bash
python main.py --devices                    # inventaire des périphériques
python main.py --wav-in essai.wav           # rejoue un WAV 16 bits mono, un tour
python main.py --wav-out reponse.wav        # écrit la parole au lieu de la jouer
python main.py --stt scripted --tts silent  # ni transcription ni voix réelles
python main.py --muet                       # ne joue rien : mesure de latence pure
```

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
| `@skill(confirm="…")` | Action irréversible : Capucine pose la question et attend un oui avant d'exécuter. |
| `@skill(isolate=True)` | Exécute dans un sous-processus, réellement tuable. Voir la limite assumée plus bas. |
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

### Les quatre plugins livrés

Ce sont de la documentation vivante, chacun couvrant un cas différent :

| Fichier | Ce qu'il montre |
|---|---|
| `heure.py` | Le contrat minimal, et le retour `{"speak", "display"}` : « neuf heures vingt » se dit, `09:20` se journalise. |
| `minuteur.py` | De l'état en mémoire, `announce()` pour interrompre depuis une tâche de fond, et `on_unload()` qui annule les minuteries — sans lui, chaque rechargement laisserait des orphelines. |
| `notes.py` | `data_dir()`, `CONFIG_DEFAULTS`, et `confirm=` sur l'effacement. |
| `systeme.py` | Une dépendance optionnelle (`psutil`) traitée par dégradation plutôt que par échec, et du code qui diffère selon la plateforme sans que le cœur en sache rien. |

---

## Architecture

```
IDLE ──« Capucine »──▶ WAKE ─▶ LISTEN ─▶ TRANSCRIBE ─▶ THINK
                                                         │
IDLE ◀── suivi expiré ──── SPEAK ◀───── ACT ◀────────────┘
  ▲                          │
  └──── barge-in ────▶ LISTEN
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
    ├── listener.py         le fil qui tient le micro, du début à la fin
    ├── endpointer.py       fin d'énoncé, pré-roll, détection de barge-in
    ├── audio.py            un seul point d'entrée/sortie, sans dépendance
    ├── registry.py         découverte, chargement, isolation des pannes
    ├── router.py           choix de l'outil, trois étages
    ├── schema.py           signature Python → JSON Schema
    ├── plugin.py           @skill et contrat public
    ├── config.py           TOML en couches
    ├── conversation.py     persona + mémoire courte
    ├── downloads.py        récupération explicite des poids
    ├── text.py             normalisation, nombres français, découpage en phrases
    ├── interfaces/         ABC seulement, aucune dépendance lourde
    └── engines/            implémentations, importées paresseusement
        ├── llm/            mock, ollama, llamacpp
        ├── stt/            faster-whisper, vosk, scripted
        ├── tts/            piper, silent
        ├── vad/            silero (onnxruntime), énergie, scripted
        └── wake/           openwakeword, vosk, scripted
tools/entrainer_capucine.py entraînement du modèle de mot d'éveil
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

### La chaîne vocale

Le transport audio parle en **PCM 16 bits mono** de bout en bout — le format
natif de `sounddevice` comme de Piper. La conversion en flottants n'a lieu qu'à
la frontière de Whisper, qui dépend de numpy de toute façon. `core/audio.py`
n'a donc aucune dépendance et s'importe sur une machine nue ; `sounddevice`
n'est chargé qu'à l'ouverture d'un flux réel.

**La synthèse est diffusée phrase par phrase.** Dès qu'une phrase est complète
dans le flux du modèle, elle part à Piper puis au haut-parleur, pendant que le
modèle écrit la suivante. Le découpage évite de couper « M. Dupont » ou
« 3.5 », et chaque phrase est un point d'interruption propre — c'est ce sur
quoi le barge-in de l'étape 3 viendra se brancher.

**Whisper invente des phrases sur du silence** — « Sous-titrage Société
Radio-Canada » est la plus célèbre. Avec un assistant déclenché à la voix, cela
donne des commandes fantômes. On ne l'appelle donc pas en dessous d'un certain
niveau sonore, on écarte une courte liste de formules connues, et on désactive
`condition_on_previous_text`, qui fait boucler le modèle sur des énoncés
courts.

### Le rechargement à chaud

Trois précautions, dont aucune ne se devine :

* **Anti-rebond.** Un éditeur ne produit pas un événement par sauvegarde mais
  trois ou quatre : fichier temporaire, renommage atomique, changement de
  droits. Sans regroupement on rechargerait quatre fois — et la première sur
  un fichier tronqué.
* **Empreinte du contenu.** Un formateur, un `touch`, une synchronisation
  réécrivent un fichier à l'identique. On compare le hachage : pas de
  changement, pas de rechargement, pas d'annonce intempestive.
* **Compilation directe de la source.** CPython valide son cache de bytecode
  sur *(date de modification, taille)*. Remplacer « Bonjour » par « Bonsoir »
  dans la même seconde ne change ni l'une ni l'autre : le rechargement
  rejouerait silencieusement l'ancien code. Le registre compile la source
  lui-même, ce qui supprime le problème et évite d'écrire des `.pyc` dans le
  dossier des plugins.

L'annonce vocale est volontairement discrète : on n'annonce qu'un nom de
compétence *nouveau*. Pendant le développement, on enregistre un fichier trente
fois par heure ; une Capucine qui commente chaque sauvegarde devient vite
insupportable.

### L'écoute permanente

**Un seul thread tient le micro**, ouvert du début à la fin de la session.
C'est ce qui rend le barge-in possible : le micro n'est jamais fermé, même
pendant que Capucine parle. Selon le mode courant, chaque trame part vers le
détecteur de mot d'éveil, vers le découpeur d'énoncé, ou vers la surveillance
d'interruption. Le thread ne décide de rien : il émet des événements que la
boucle asyncio consomme.

**Terminer une phrase sans couper l'utilisateur** demande trois précautions :
un silence exigé plus long qu'on ne le croit (700 ms), un **pré-roll** qui
conserve l'audio *précédant* la détection — sans quoi la première syllabe est
perdue — et une durée minimale de parole, pour qu'une porte qui claque ne
devienne pas une commande.

**Le barge-in doit composer avec l'écho.** Sans annulation acoustique, le
micro entend le haut-parleur et Capucine se coupe elle-même. Trois garde-fous
réglables : un seuil plus haut que pour l'écoute normale (0,85), un délai de
garde au début de la réponse, et une durée de parole soutenue exigée. Au
casque, on peut tout abaisser. Sur haut-parleur ouvert — le cas d'un Pi —
`barge_in.mode = "eveil"` (n'interrompre que si l'on redit « Capucine ») reste
le plus sûr ; c'est le défaut du profil Pi.

**Le mode suivi** garde l'écoute ouverte quelques secondes après la réponse :
on enchaîne sans redire le nom. `assistant.follow_up_seconds = 0` le désactive.

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

## Mesurer et régler

```bash
python tools/mesurer_latence.py                  # tout ce qui est disponible
python tools/mesurer_latence.py --profile pi
python tools/mesurer_latence.py --wav essai.wav  # transcrire un vrai enregistrement
python tools/mesurer_latence.py --json           # pour comparer deux machines
```

Les deux familles de chiffres ne se lisent pas de la même façon.

**Les étages permanents** — mot d'éveil et VAD — tournent en continu, du
démarrage à l'extinction. Ce qui compte pour eux n'est pas la latence mais le
**facteur temps réel** : le rapport entre le temps de calcul et la durée
d'audio traitée. À 0,10, un cœur sur dix est occupé en permanence ; au-dessus
de 0,5, le Pi n'aura plus de souffle pour transcrire. Le banc additionne les
deux et vous dit combien d'un cœur part en fond de tâche.

**Les étages à la demande** — transcription, routage, plugin, synthèse — ne
coûtent que pendant un tour. Là c'est la latence qui compte, et surtout le
**temps avant la première parole** : c'est lui que l'utilisateur ressent, pas
la durée totale de la réponse.

En cours de session, `/latences` donne la médiane, le p90 et le maximum par
étage sur les derniers tours, et `/machine` relit la configuration à la lumière
du matériel détecté.

> Les chiffres dépendent tellement de la machine que ce dépôt n'en publie
> aucun comme référence. Mesurez les vôtres : c'est une commande, et c'est la
> seule façon honnête de savoir si votre carte suit.

## Tests

```bash
python -m pytest
```

278 tests, aucun modèle téléchargé, aucun périphérique audio requis. Le critère
d'acceptation est joué en entier — vrai observateur `watchdog`, vrai fichier
déposé pendant l'exécution. La boucle vocale — éveil, énoncé, réponse, suivi,
barge-in — est éprouvée avec un micro en mémoire, un mot d'éveil scripté et un
VAD scripté. Les adaptateurs Piper et
faster-whisper sont en plus vérifiés contre les **signatures réelles** des
bibliothèques installées, de sorte qu'une dérive d'API fasse échouer la suite.

---

## Feuille de route

| Étape | Contenu | État |
|---|---|---|
| 1 | Squelette, config, interfaces, registre de plugins, routeur, mode texte | **fait** |
| 2 | STT (`faster-whisper`, Vosk) + TTS (`piper`), pipeline vocal au clavier | **fait** |
| 3 | Mot d'éveil « Capucine » (`openWakeWord` + repli Vosk), VAD (`silero`), barge-in, mode suivi | **fait** |
| 4 | Rechargement à chaud (`watchdog`), isolation en sous-processus, confirmation, quatre plugins d'exemple | **fait** |
| 5 | Profil Raspberry Pi, mesures de latence, guide d'installation par plateforme | **fait** |

### Entraîner le mot d'éveil

```bash
python tools/entrainer_capucine.py preparer      # config + état des prérequis
python tools/entrainer_capucine.py echantillons  # positifs français, voix Piper
python tools/entrainer_capucine.py entrainer     # pipeline officiel openWakeWord
python tools/entrainer_capucine.py installer     # copie le modèle dans models/wake/
python tools/entrainer_capucine.py essayer a.wav # vérifie sur un enregistrement
```

`preparer` liste ce qui manque et la commande pour l'obtenir : le générateur
d'échantillons Piper, des réponses impulsionnelles, des bruits de fond, torch.
Ces deux derniers ne sont pas facultatifs — sans eux, le modèle apprend une
pièce et un micro, pas un mot.

Une mise en garde honnête : le pipeline officiel d'openWakeWord repose sur une
génération de données synthétiques pensée pour l'anglais. Pour un mot français,
la qualité des positifs est le facteur limitant, d'où la sous-commande
`echantillons`, qui les produit avec les **voix françaises de Piper** en variant
débit, couleur et niveau. Attendez-vous à ce que le repli Vosk reste le chemin
principal un certain temps.

### La limite annoncée à l'étape 1, et sa parade

On ne peut pas tuer un thread en Python. Un plugin parti en boucle infinie est
*abandonné* — Capucine répond et le met en quarantaine — mais son thread
continue jusqu'à ce qu'il finisse.

La parade existe désormais, en option et par compétence :

```python
@skill(description="…", isolate=True, timeout=5)
def analyse_lourde(fichier: str) -> str:
    ...
```

Le sous-processus est réellement tuable. Le prix est réel aussi, et c'est
pourquoi ce n'est pas le défaut : 100 à 300 ms de démarrage par appel, des
arguments et un retour qui doivent être sérialisables, et aucun état conservé
entre deux appels — un minuteur ne peut pas s'isoler. À réserver aux
compétences qui manipulent des données douteuses ou des bibliothèques natives
capables de bloquer indéfiniment.

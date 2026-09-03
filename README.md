# Lily

Assistante vocale francophone, **entièrement locale**. Aucun service tiers,
aucune clé d'API, aucun appel réseau sortant : elle fonctionne le Wi-Fi coupé.

La contrainte qui prime sur toutes les autres décisions d'architecture :
**ajouter une capacité, c'est déposer un fichier Python dans `plugins/`.** Sans
redémarrer l'assistante, et sans jamais toucher au cœur.

> **État : les cinq étapes sont terminées**, plus une extension d'assistance
> (mémoire persistante, recherche web, fichiers, Python, projets). Le critère
> d'acceptation du projet passe : déposez un fichier dans `plugins/` pendant
> que Lily tourne, elle annonce la nouvelle compétence et l'exécute — sans
> redémarrage et sans toucher au cœur.

---

## Le critère d'acceptation, en trois gestes

Lily tourne. Sans l'arrêter, sans toucher à une ligne du cœur :

```bash
cat > plugins/dés.py <<'EOF'
import random
from lily.plugin import skill

@skill(description="Lance un dé.", examples=["lance un dé", "tire un dé"])
def lancer_de(faces: int = 6) -> str:
    """Lance un dé à N faces."""
    return str(random.randint(1, faces))
EOF
```

```
Lily › Nouvelle compétence disponible : lancer de.
Vous  › lance un dé à vingt faces
Lily › 13
```

Aucun modèle de langage n'a été sollicité : l'étage déterministe du routeur a
reconnu l'outil sur ses `examples` et lu « vingt » comme `faces=20`. Le
scénario est rejoué en entier par `tests/test_recette.py`, avec un vrai
observateur de fichiers.

---

## Installation

Python 3.11 ou plus récent. Le cœur ne dépend que de la bibliothèque standard ;
tout le reste est optionnel et chargé paresseusement. On installe donc par
couches, et Lily fonctionne à chaque étape.

Rien n'est jamais téléchargé automatiquement au démarrage : un assistant censé
fonctionner le Wi-Fi coupé ne sort pas sur le réseau sans qu'on le lui demande.

### Windows

> Le dépôt GitHub porte encore son ancien nom : `git` suit la redirection le
> jour où vous le renommerez, la commande ci-dessous marche dans les deux cas.

```powershell
git clone https://github.com/tbouillaguet-max/capucine.git Lily
cd Lily
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,reload]"

# Essayez tout de suite, sans le moindre modèle :
python main.py --text --llm mock
```

**Le modèle de langage.** Ollama s'installe *hors* de l'environnement Python —
c'est un service Windows, pas un paquet pip :

```powershell
winget install --id Ollama.Ollama
#  ou le programme d'installation : https://ollama.com/download/windows
```

**Fermez puis rouvrez PowerShell**, sans quoi la commande reste introuvable :
le `PATH` d'un terminal déjà ouvert n'est pas mis à jour par l'installation.
`ollama : The term 'ollama' is not recognized` veut dire l'un ou l'autre —
pas installé, ou terminal pas rouvert.

```powershell
ollama --version                           # doit répondre
pip install -e ".[llm-ollama]"
ollama pull qwen2.5:7b-instruct-q4_K_M     # ~4,7 Go, une fois
ollama pull nomic-embed-text               # 274 Mo, pour la recherche par le sens
python main.py --text
```

Pas de `ollama serve` à lancer sous Windows : l'installateur enregistre le
service, qui démarre tout seul et vit dans la zone de notification. Le lancer
à la main donnerait « Only one usage of each socket address ». C'est sous
Linux qu'il faut parfois le démarrer soi-même.

En attendant, `python main.py --text --llm mock` fonctionne : l'étage
déterministe appelle les compétences, seule la conversation libre manque.

**La voix.** `sounddevice` embarque PortAudio dans ses roues Windows : rien à
installer à côté.

```powershell
pip install -e ".[audio]"
python -m lily.core.downloads tout     # voix Piper + modèle Whisper
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
python -m lily.core.downloads vosk
python main.py                             # dites « Lily »
```

### Linux et Raspberry Pi

```bash
sudo apt update
sudo apt install python3 python3-venv python3-dev git libportaudio2

git clone https://github.com/tbouillaguet-max/capucine.git Lily
cd Lily
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
ollama pull nomic-embed-text               # 274 Mo, pour la recherche par le sens
```

Le script d'installation pose un service systemd et le démarre. S'il ne
répond pas — `systemctl status ollama` —, `ollama serve` dans un autre
terminal fait l'affaire le temps d'un essai.

**La voix et l'écoute.**

```bash
pip install -e ".[audio,wake,vosk]"
pip install --no-deps silero-vad
python -m lily.core.downloads tout --profile pi
python -m lily.core.downloads vosk

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
sudo cp deploy/lily.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now lily
journalctl -u lily -f
```

Le fichier est commenté et à adapter : utilisateur, chemins, périphérique
audio. Il journalise en JSON, ce qui permet d'agréger les latences ailleurs.

### Pourquoi `pip install --no-deps silero-vad`

Le paquet `silero-vad` importe `torch` **et** `torchaudio` dès son `__init__`,
y compris sur le chemin ONNX : plusieurs centaines de méga-octets sur un Pi,
pour un modèle de moins d'un méga-octet. Or le fichier `silero_vad.onnx` est
livré *dans* le paquet. Lily le localise sans importer le paquet et
l'exécute avec `onnxruntime` : mêmes poids, sans la chaîne torch.

Sans lui, un VAD par énergie à plancher de bruit adaptatif prend le relais
automatiquement — moins fin dans le bruit, mais opérationnel partout.

### Le mot d'éveil demande un modèle qui n'existe pas encore

openWakeWord ne fournit aucun modèle « lily » pré-entraîné ; il faut
l'entraîner, ce que pilote `tools/entrainer_lily.py`. En attendant,
Lily bascule automatiquement sur le repli Vosk à grammaire restreinte —
un décodeur qui n'a le droit de reconnaître que « lily » et rien d'autre,
donc rapide et peu gourmand. Ce n'est pas une panne, c'est un état normal du
projet.

Conséquence pour le corpus d'éveil : `entrainer_lily.py corpus` marche
dès maintenant (il ne fait que ranger des fichiers), mais `seuil` réclame un
modèle openWakeWord installé — il n'a donc pas pu être exécuté de bout en
bout ici, faute de modèle « lily » à mesurer.

## Utilisation

```bash
python main.py                              # écoute permanente : dites « Lily »
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
from lily.plugin import skill

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
| `@skill(timeout=…)` | Délai maximum. Au-delà, Lily répond qu'elle n'a pas pu exécuter la commande. |
| `@skill(confirm="…")` | Action irréversible : Lily pose la question et attend un oui avant d'exécuter. |
| `@skill(isolate=True)` | Exécute dans un sous-processus, réellement tuable. Voir la limite assumée plus bas. |
| `raise SkillRefused("…")` | Refuser **en le disant**. Le message est prononcé mot pour mot, ce n'est pas compté comme une panne. |
| `demander_au_modele(prompt)` | Une complétion du modèle local. Pas de routage, donc pas de récursion : un plugin ne peut pas en déclencher un autre par ce biais. |
| `atelier()` | Les dossiers que vous avez ouverts. Tout accès disque devrait passer par là. |
| `memoire()` | L'historique et les faits durables. |
| `conversation()` | Le fil courant, pour reprendre une session passée. |
| Docstring | Contexte pour le modèle. Les sections `Args:` ou `:param x:` documentent chaque argument. |
| `CONFIG_DEFAULTS = {…}` | Réglages du plugin, surchargés par `[plugins.<nom>]` du fichier de configuration. |
| `get_config("cle")` | Lit ces réglages. Utilisable dès le corps du module. |
| `data_dir()` | Dossier inscriptible réservé au plugin, créé à la demande. |
| `get_logger()` | Journal nommé d'après le plugin. |
| `announce("…")` | Fait parler Lily hors d'un tour — pour une tâche de fond, un minuteur qui sonne. |
| `on_load()` / `on_unload()` | Cycle de vie : ouvrir une connexion, libérer une ressource. |

### Ce que le plugin retourne

* une **chaîne** → lue à voix haute telle quelle ;
* un **dict** `{"speak": …, "display": …}` → dissocie ce qui est dit de ce qui
  est journalisé ;
* `None` → Lily ne dit rien.

### Règles

* **Jamais d'installation automatique de dépendance.** Un import manquant
  écarte le plugin avec un message qui nomme le paquet et la commande.
* **Un plugin ne fait jamais tomber Lily.** Erreur à l'import, exception à
  l'exécution, dépassement de délai, `sys.exit()` : tout est confiné.
* Un plugin en échec répété est mis en quarantaine et retiré du catalogue
  proposé au modèle.

### Les plugins livrés

Quatre plugins **pédagogiques**, documentation vivante du contrat :

| Fichier | Ce qu'il montre |
|---|---|
| `heure.py` | Le contrat minimal, et le retour `{"speak", "display"}` : « neuf heures vingt » se dit, `09:20` se journalise. |
| `minuteur.py` | De l'état en mémoire, `announce()` pour interrompre depuis une tâche de fond, et `on_unload()` qui annule les minuteries — sans lui, chaque rechargement laisserait des orphelines. |
| `notes.py` | `data_dir()`, `CONFIG_DEFAULTS`, et `confirm=` sur l'effacement. |
| `systeme.py` | Une dépendance optionnelle (`psutil`) traitée par dégradation plutôt que par échec, et du code qui diffère selon la plateforme sans que le cœur en sache rien. |

Six plugins d'**assistance**, qui font vraiment travailler Lily :

| Fichier | Ce qu'elle sait faire |
|---|---|
| `memoire.py` | Retenir un fait durablement, retrouver un passage d'une conversation passée, en reprendre une. |
| `recherche.py` | Chercher sur le web et lire une page. **Le seul plugin qui sort de la machine.** |
| `fichiers.py` | Lister, lire, chercher, compléter, écrire, déplacer, jeter — dans un périmètre que vous ouvrez. |
| `python.py` | Exécuter du Python, écrire du code **en voyant les signatures de vos fonctions**, et boucler jusqu'à ce qu'il tourne. |
| `documents.py` | Ouvrir Word, Excel, PowerPoint, PDF et CSV : lire, résumer, chercher à travers, **indexer**. |
| `projet.py` | Lancer un dépôt entier en tâche de fond, suivre son avancement, lire son rapport de run, jouer ses tests. |
| `calculrisque.py` | Piloter CalculRisque_Mark5 : mises à jour, optimisations de paramètres, création de stratégies. |

Et un plugin d'**introspection** :

| Fichier | Ce qu'elle sait faire |
|---|---|
| `apprentissage.py` | Montrer ce qu'elle a retenu de votre façon de parler, dicter un mot à son vocabulaire, tout lui faire oublier. |
| `connaissances.py` | Interroger ce qu'elle a lu — vos documents, vos conversations — et répondre en citant d'où ça vient. |
| `routines.py` | Retenir un enchaînement que vous venez de faire, et **s'écrire le plugin correspondant**. |

---

## Ce qu'elle sait faire pour vous

Ces capacités dépassent le cadre d'un assistant vocal ordinaire. Elles sont
donc encadrées, et il faut lire cette section avant de les ouvrir.

### La mémoire

```
Vous  › retiens que je travaille sur CalculRisque
Lily › C'est retenu.
                       ⟶ redémarrage ⟵
Vous  › que sais-tu de moi
Lily › Tu travailles sur CalculRisque.
Vous  › reprends notre conversation d'hier
Lily › Nous reprenons la conversation d'hier à 18 h 40 : le backtest options.
```

Trois horizons distincts, souvent confondus : le **fil courant** (borné, il
tient dans le contexte du modèle), l'**historique** (toutes les conversations,
cherchables et reprenables) et les **faits durables** (réinjectés dans le
persona à chaque tour). Le tout dans un fichier SQLite — bibliothèque
standard, aucune dépendance ajoutée — sur votre machine, qui n'en sort jamais.

En session : `/conversations`, `/reprendre [n]`, `/memoire`. Au lancement :
`python main.py --reprendre derniere`.

### L'atelier — le périmètre sur vos fichiers

**Il est vide par défaut, et ce n'est pas un oubli.** La commande arrive par la
voix, une transcription est imparfaite, et un modèle 7B choisit parfois mal ses
arguments. Les compétences fichiers, Python et projet restent inertes tant que
vous n'avez pas ouvert un dossier :

```toml
[atelier]
racines = ["~/projets/CalculRisque_Mark5"]
```

ou, le temps d'une session : `python main.py --atelier ~/projets/CalculRisque_Mark5`.

Ce que la frontière garantit, et qui est éprouvé par `tests/test_atelier.py` :

* tout chemin est résolu — **liens symboliques compris** — puis vérifié comme
  appartenant à une racine autorisée ; un lien posé dans l'atelier ne donne pas
  accès à sa cible ;
* les identifiants et les clés (`.env`, `*.pem`, `id_rsa`, `.ssh/`) restent
  hors de portée **même à l'intérieur** d'une racine ouverte ;
* elle **refuse d'écrire du texte dans un fichier binaire** : un `.xlsx`
  réécrit en UTF-8 ne serait pas modifié, il serait détruit. Le verdict porte
  sur le contenu quand le fichier existe, sur l'extension sinon ;
* toute réécriture laisse une sauvegarde horodatée à côté du fichier ;
* **rien n'est supprimé** — les fichiers partent à la corbeille ;
* écrire, déplacer, jeter, exécuter du code et lancer un projet demandent
  confirmation à voix haute ;
* `atelier.lecture_seule = true` si vous voulez qu'elle regarde sans toucher.

Un refus est **prononcé**, pas avalé : « ce fichier est hors de l'atelier »
plutôt que « je n'ai pas pu exécuter cette commande ».

### Les documents que l'UTF-8 ne sait pas lire

Un `.docx` ou un `.xlsx` n'est pas du texte : c'est une archive de XML. Ces
formats passent par les bibliothèques dédiées, importées **format par format**
— un `.docx` reste lisible même si `openpyxl` manque.

```bash
pip install -e ".[documents]"
```

```
Vous  › ouvre le rapport word
Lily › Rapport trimestriel de valorisation. Le multiple médian ressort à 12,4x.
Vous  › quelles feuilles il y a dans le classeur
Lily › 2 feuilles : Synthèse, Détail.
Vous  › cherche le mot médian dans mes documents
Lily › 2 documents : budget.xlsx, rapport.docx.
```

Ce qu'elle lit : Word (paragraphes **et** tableaux — ils portent souvent
l'essentiel), Excel (valeurs plutôt que formules, feuille par feuille),
PowerPoint (diapositives **et** notes du présentateur), PDF, CSV et TSV.

Deux limites franches. Les anciens formats binaires `.doc`, `.xls`, `.ppt` ne
sont pas lus — elle vous demande de réenregistrer en `.docx`. Et une formule
qu'Excel n'a jamais calculée apparaît vide : c'est le résultat mis en cache
qui est lu, rien ne recalcule à la place d'Excel.

**Lecture seulement.** Écrire dans un document Office demande de préserver
styles, formules et mises en page ; le faire à moitié abîmerait vos fichiers.

### Coder et exécuter

```
Vous  › écris-moi un script qui trie un csv par date
Lily › J'ai écrit 18 lignes. Relisez-les, puis dites-moi où les enregistrer.
          [le code s'affiche]
Vous  › enregistre-le dans outils/tri.py
Lily › Voulez-vous vraiment enregistrer ce code ?
```

Le cycle est **proposer puis enregistrer**, en deux compétences : un modèle 7B
écrit du Python approximatif, on ne l'écrit jamais à l'aveugle. L'exécution se
fait en sous-processus, avec délai, dans l'atelier, sans jamais passer par un
shell.

### Lancer un projet entier

Un pipeline de données tourne quarante minutes : hors de question de bloquer un
tour de parole. Lily lance, rend la main tout de suite, et vous interrompt
à la fin.

```toml
[plugins.projet.projets.calculrisque]
chemin = "~/projets/CalculRisque_Mark5"
description = "valorisation et backtest options"
commande = "run_pipeline_quarterly.py"
options_par_defaut = "--resume"
rapport = "data/pipeline_runs/*/report.json"
commande_test = "-m pytest -q"
delai_s = 5400
variables = { SEC_CONTACT_EMAIL = "vous@exemple.fr" }
```

```
Vous  › lance le pipeline calculrisque
Lily › C'est parti pour calculrisque. Je vous préviens à la fin.
Vous  › où en est le pipeline
Lily › calculrisque tourne depuis 12 minutes. 06b_calcul_valorisation_combinee.py
                       ⟶ trente minutes plus tard ⟵
Lily › calculrisque est terminé, en 41 minutes.
```

`rapport` pointe vers le JSON de run : Lily sait alors dire « le dernier run
s'est terminé en partial, étapes en échec : 08_recuperation_options.py ».
Ajouter un projet ne demande pas de toucher au code.

### La recherche web — la seule entorse

**Ce plugin sort de la machine.** C'est la seule dérogation à la règle numéro
un du projet, et elle est délibérée : elle vit dans un plugin, pas dans le
cœur. Retirez `plugins/recherche.py` et Lily redevient intégralement
hors-ligne. Trois moteurs :

| Moteur | Ce qu'il demande | Remarque |
|---|---|---|
| `searxng` | une instance que vous hébergez | Le choix cohérent : il interroge Google et les autres, mais depuis chez vous, sans compte ni profilage. |
| `google` | une clé d'API et un identifiant de moteur | L'API officielle *Custom Search*, 100 requêtes par jour gratuites. |
| `duckduckgo` | rien | Sans garantie : c'est du décorticage de page HTML, ça casse le jour où la mise en page change. |

Elle annonce qu'elle va sur le réseau plutôt que de le faire en silence, et
hors-ligne elle le dit au lieu de planter.

### Coder sur VOTRE dépôt : trois leviers, et de quoi les mesurer

Le mode d'échec numéro un d'un petit modèle sur un dépôt qu'il ne connaît pas
n'est pas la syntaxe, c'est l'**invention d'API**. Il ignore que
`capped_weights(conviction, cap_pct=None)` existe, alors il écrit
`normalize_weights()` — plausible, inexistante, et rien ne tourne.

```
Vous  › écris une fonction qui pondère des convictions en plafonnant les poids
Lily › J'ai écrit 12 lignes, en voyant 2 de vos fonctions.
                                     ⟶ affiché ⟵
             def backtest.strategies.base.capped_weights(conviction: pd.Series,
                 cap_pct: float | None = None, max_iter: int = 20) -> pd.Series
                 → Poids proportionnels à `conviction`, aucun ne dépassant
                   cap_pct % du portefeuille.
```

**1. Le catalogue d'API** (`lily/core/catalogue.py`). Il lit vos signatures
et vos docstrings **par AST, jamais par import** — aucun effet de bord, et pas
besoin que les dépendances du dépôt soient installées. C'est le mécanisme de
`schema.py` (signature + docstring = contrat) appliqué au code que vous
écrivez, et plus seulement aux compétences.

Il ne déverse pas : il **sélectionne**. Injecter douze fonctions au hasard dans
le contexte d'un 7B est pire que n'en injecter aucune, donc une demande sans
rapport ne rend **rien**. Le tri se fait sur la forme de la distribution — une
demande qui a sa réponse dans le dépôt produit un pic, une demande hors sujet
un plateau — avec un plancher mesuré, pas deviné (voir le commentaire dans le
code, et refaites la mesure si vous changez le score).

Vos identifiants sont en anglais, vos docstrings et vos demandes en français :
l'appariement se fait donc aussi **par cognat**, et « volatilité réalisée »
retrouve `realized_volatility`. Limite assumée : « pricer » ne trouve pas
`pricing`, les deux ne partagent pas assez de préfixe.

**2. La boucle de vérification** — « écris ça et teste-le ». Elle écrit,
exécute, lit la trace d'erreur, corrige, et recommence, bornée en tentatives.
C'est le levier le plus sous-estimé et il ne coûte pas un octet de GPU : un
modèle qui voit sa propre trace transforme un pari en itération.

Ce qu'elle **n'établit pas** : que le résultat soit juste. Elle prouve que le
code s'exécute sans lever, ce qui n'est pas la même chose — un calcul faux qui
ne plante pas passe ce filtre.

**3. Le modèle.** `ollama pull qwen2.5-coder:7b`, puis `llm.model` dans la
configuration. Même taille que le généraliste, entraîné pour le code.

#### Et le banc, pour ne croire personne sur parole

```bash
python tools/banc_de_code.py --atelier ~/projets/CalculRisque_Mark5 --variantes
```

Il joue les mêmes tâches dans les quatre combinaisons catalogue × boucle et
**note en exécutant** : le code produit et sa vérification tournent ensemble,
et la tâche passe si le processus rend zéro. Aucun modèle-juge, aucune
appréciation — la même récompense vérifiable qui a fait progresser les modèles
de code.

Les tâches sont dans `tools/banc/taches.toml`, en deux familles : **logique**
(la boucle se voit là) et **api** — celles-là exigent que le code appelle
vraiment `capped_weights` ou `bs_price`, et c'est là que le catalogue se voit.
Ajoutez les vôtres : un banc n'a de valeur que s'il ressemble à ce que vous
demandez vraiment. Lisez les colonnes plutôt que le total — un levier qui ne
bouge pas sa colonne ne paye pas.


### Comment elle apprend — les six mécanismes

Aucun ne réentraîne un modèle de langage, et c'est un choix plutôt qu'un
renoncement : affiner un 7B demanderait des heures de GPU et des milliers
d'exemples que personne n'a, pour un gain que ceux-là obtiennent dès le
deuxième tour. Tout tient dans le SQLite de la mémoire, quelques fichiers
audio, et un plugin qu'elle s'écrit.

| Ce qui s'apprend | Comment | Où c'est décrit |
|---|---|---|
| **Vos formulations** | l'étage déterministe rate, le modèle tranche juste, la phrase est retenue | [Ce qu'elle apprend de vous](#ce-quelle-apprend-de-vous) |
| **Vos corrections** | « non, je voulais dire… » désapprend le faux et apprend le juste | idem |
| **Votre vocabulaire** | vos noms propres sont soufflés à la transcription | idem |
| **Votre voix** | les extraits d'éveil sont gardés et étiquetés tout seuls | [Ce qu'elle apprend de votre voix](#ce-quelle-apprend-de-votre-voix) |
| **Vos documents** | indexés, cherchés par le sens, cités quand elle répond | [Ce qu'elle a lu](#ce-quelle-a-lu--vos-documents-interrogeables) |
| **Vos enchaînements** | montrés une fois, elle en écrit un plugin | [Les routines](#les-routines--elle-écrit-son-propre-plugin) |

Chacun se coupe séparément dans la configuration, et deux d'entre eux — le
corpus sonore et l'index des documents — sont explicitement encadrés parce
qu'ils touchent à ce que vous lui confiez.


### Piloter un projet qu'elle connaît vraiment

`projet.py` lance **n'importe quel** projet et rend compte. `calculrisque.py`
en sait davantage sur **celui-là** : ses enchaînements nommés, ses cinq
optimiseurs, et la forme exacte d'une stratégie dans son registre.

```
Vous  › fais la mise à jour quotidienne
Lily › C'est parti : mise à jour quotidienne, 1 étape.
Vous  › optimise le seuil d'entrée
Lily › C'est parti. Ça prend des heures ; je vous préviens à la fin.
                              ⟶ trois heures plus tard ⟵
Lily › optimisation seuil_entree : terminé, 1 étape réussie.
Vous  › qu'est-ce que ça a donné
Lily › D'après seuil_entree_20260830.csv, le meilleur réglage est :
           entry_threshold_pct 35, sharpe 1.42.
```

Trois choses, et les trois garde-fous qui vont avec :

| Ce qu'elle sait faire | Le garde-fou |
|---|---|
| **Enchaîner** les étapes d'une mise à jour, en tâche de fond, journal lisible à la voix | un seul travail à la fois, et la chaîne s'arrête à la première étape qui échoue plutôt que de bâtir sur du faux |
| **Optimiser** les paramètres (stops, rebalancement, convergence, seuil d'entrée, multiples) | elle **lit la source du script** pour savoir quels drapeaux il accepte : les cinq optimiseurs n'ont pas le même jeu d'options, et un drapeau inconnu ferait échouer argparse avant tout calcul |
| **Créer une stratégie** dans `backtest/strategies/`, l'enregistrer, prouver qu'elle charge | **gabarit vérifié, pas de code écrit par le modèle** |

**Le modèle n'écrit pas les stratégies.** Un 7B quantifié qui rédige de la
logique de valorisation produit du code plausible et silencieusement faux — et
il s'agit d'argent. Le gabarit reprend la mécanique de `valuation_gap_dcf`
(écart corrigé de l'inflation, poids plafonnés par `capped_weights`) et n'y
injecte que vos réglages : seuil d'entrée, conviction sur l'écart brut ou sur
l'excédent sectoriel, pondération proportionnelle ou égale, plafond de
positions, filtre de secteurs. Le fichier produit est compilé avant d'être
écrit, puis Lily interroge le **vrai registre** (`09_backtest.py
--list-strategies`) pour vérifier qu'il charge — écrire un fichier qui ne
s'importe pas serait pire que ne rien écrire.

**Elle n'écrit jamais dans un fichier existant du dépôt**, à une exception
près : la seule ligne d'import qui enregistre la nouvelle stratégie dans
`backtest/strategies/__init__.py`, avec sauvegarde par l'atelier. Ni les
scripts de valorisation, ni le moteur de backtest, ni `config.py` ne sont
touchés. Et si ce fichier n'a pas la forme attendue, elle ne le retouche pas à
l'aveugle : elle vous dit la ligne à ajouter.

Tout passe par l'atelier : tant que vous n'avez pas ouvert le dossier du
projet, ces compétences refusent. Les enchaînements et les optimiseurs sont
déclarés dans `config/pc.toml` — ce sont des **données**, en ajouter un ne
demande pas de toucher au plugin.


### Ce qu'elle a lu — vos documents, interrogeables

```
Vous  › indexe tout le dossier des rapports
Lily › J'ai indexé 14 documents, et sauté 3 inchangés.
Vous  › d'après mes documents, combien on a perdu au premier trimestre
Lily › Une perte de douze mille quatre cents euros, d'après le rapport Q1.
                                        ⟶ affiché ⟵
                       Sources :
                       · rapport_q1.docx  (proximité 0.81)
                       · budget_2026.xlsx — feuille Synthèse  (proximité 0.64)
```

Elle ne cherche pas des mots, elle cherche du **sens** : « combien on a perdu
au premier trimestre » retrouve « la perte de Q1 s'élève à », sans un mot en
commun. Puis elle répond à partir des passages retrouvés — et les nomme, ce
qui permet de vérifier.

Trois refus assumés dans cette conception :

- **Rien ne sort de la machine.** Le vectoriseur est local (Ollama en
  bouclage, l'hôte est validé comme pour le dialogue) : vectoriser un
  document, c'est en envoyer le contenu au moteur.
- **Aucune dépendance nouvelle.** Les vecteurs sont des BLOB dans le SQLite
  de la mémoire. Pas de base vectorielle à installer, pas de service à tenir
  en vie. La recherche est un balayage : mesuré ici, 171 ms pour 5 000
  fragments en Python pur, 11 ms si numpy est là. Au-delà de
  `fragments_max`, elle le dit au lieu de ralentir en silence.
- **Sans modèle de plongement, ça marche quand même** — en plein texte (FTS5).
  Moins fin, jamais absent, et elle annonce sur quel mode elle tourne au lieu
  de laisser croire à une recherche par le sens.

Pour la recherche par le sens : `ollama pull nomic-embed-text` (274 Mo). Elle
indexe aussi **ses propres conversations**, au fil des tours, dans un fil de
fond : ce que vous lui avez dit la semaine dernière redevient trouvable.
Réindexer un dossier ne refait que les fichiers modifiés, et changer de modèle
de plongement invalide l'index — deux espaces vectoriels ne se comparent pas.


### Ce qu'elle apprend de vous

```
Vous  › relance-moi le bazar du dépôt CalculRisque_Mark5
Lily › Le pipeline est parti.        (le modèle a tranché : 380 ms de réflexion)
                       ⟶ le lendemain ⟵
Vous  › relance-moi le bazar
Lily › Le pipeline est parti.        (l'étage déterministe a suffi : 4 ms)
```

Trois mécanismes, **aucun réentraînement de modèle**. C'est un choix, pas un
renoncement : affiner un 7B demanderait des heures de GPU et des milliers
d'exemples que personne n'a, pour un gain que ces trois-là obtiennent dès le
deuxième tour. Tout tient dans le fichier SQLite de la mémoire.

| Ce qui s'apprend | Quand | Ce que ça change |
|---|---|---|
| **Vos formulations** | quand l'étage déterministe rate et que le modèle tranche juste | la fois suivante, l'étage déterministe reconnaît seul — plus vite, et sans dépendre de l'humeur d'un 7B |
| **Vos corrections** | « non, je voulais dire le minuteur » | elle **désapprend** ce qui était faux *et* apprend ce qui était juste, sur la phrase d'origine — c'est elle qui reviendra, pas la correction |
| **Votre vocabulaire** | dès qu'un nom propre passe dans un tour | `CalculRisque_Mark5` est soufflé à Whisper, qui cesse d'entendre « calcul risque marque cinq » |

Les garde-fous comptent autant que les mécanismes :

- **Un tour raté n'apprend rien.** Une compétence qui échoue ou qui refuse ne
  produit aucune formulation retenue — sinon elle apprendrait ses erreurs.
- **Une formulation retenue ne dépasse jamais un exemple d'auteur.** Son poids
  monte de 0,7 à 1,0 avec les confirmations, plafonné là.
- **Une erreur se désapprend aussi vite qu'elle s'est apprise.** Un démenti de
  plus que de confirmations, et l'association disparaît.
- **Le vocabulaire est volontairement avare** : casse chameau, sigles, mots
  contenant un chiffre. Un simple mot capitalisé serait trop bruyant, et un
  faux positif dans l'amorce biaise la transcription vers des mots que vous ne
  dites jamais.
- **Rien n'est opaque** : « qu'est-ce que tu as appris », « quel vocabulaire tu
  connais », « oublie ce que tu as appris sur le minuteur ». Une mémoire qu'on
  ne peut pas inspecter est une mémoire à laquelle on ne peut pas faire
  confiance.

Chaque mécanisme se coupe séparément dans `[apprentissage]`, et
`active = false` les coupe tous.

### Ce qu'elle apprend de votre voix

Le modèle « lily » livré est entraîné sur des voix de synthèse
françaises. Il marche, mais il ne connaît ni votre timbre, ni votre débit, ni
l'acoustique de votre pièce — et c'est précisément ce qui sépare un mot
d'éveil qui répond du premier coup d'un qu'il faut répéter.

Allumé, le corpus garde les deux secondes qui **précèdent** chaque
déclenchement, et les étiquette tout seul d'après ce qui suit :

| Ce qui se passe après l'éveil | Étiquette | Pourquoi c'est précieux |
|---|---|---|
| un énoncé transcrit | **vrai positif** | votre voix, votre pièce, votre micro |
| rien du tout | **faux positif** | la télévision l'a réveillée : ces exemples-là ne s'inventent pas en studio |

```
python tools/entrainer_lily.py seuil     # mesure VOTRE seuil au lieu de le deviner
python tools/entrainer_lily.py corpus    # verse les extraits dans le jeu d'entraînement
```

`seuil` passe le modèle courant sur vos propres enregistrements et balaie les
valeurs : à chacune, combien d'éveils reconnus, combien de déclenchements
pour rien. Un faux déclenchement coupe la parole, un éveil manqué se rattrape
en répétant — le conseil pénalise donc le premier trois fois plus.

**Éteint par défaut, et ce n'est pas un oubli** (`[corpus] actif = true`) :
écrire du son sur un disque est une décision qui vous appartient. Allumé,
seule la fenêtre du mot d'éveil est gardée — jamais ce que vous dites
ensuite, garanti par deux méthodes distinctes dans le code plutôt que par une
intention — et le nombre de fichiers est borné.

### Les routines : elle écrit son propre plugin

```
Vous  › quelle heure est-il
Lily › Il est sept heures dix.
Vous  › mes notes
Lily › Vous avez deux notes…
Vous  › retiens cette routine, elle s'appelle mon matin
Lily › C'est retenu. Dites « matin » et je referai : heure, puis mes notes.
                    ⟶ plugins/routine_matin.py vient d'apparaître ⟵
Vous  › matin
Lily › Il est sept heures onze. Vous avez deux notes…
```

C'est **le critère d'acceptation du projet retourné comme un gant**. « Déposer
un fichier dans `plugins/` ajoute une capacité, sans redémarrer et sans
toucher au cœur » — ici, c'est Lily qui dépose le fichier, et elle passe
par exactement le même chemin que vous : écriture atomique, surveillance,
rechargement à chaud.

Deux choses qu'elle ne fait **pas**, délibérément :

- **Le modèle n'écrit pas ce fichier.** Le code sort d'un gabarit fixe ; du
  journal ne viennent que des noms de compétences existantes et des arguments
  passés par `json.dumps`. Un plugin généré ne peut contenir que des appels à
  ce que vous venez de faire — jamais du code inventé.
- **Elle ne contourne aucune confirmation.** Une étape déclarée `confirm=`
  arrête la routine au lieu de s'exécuter en passant : on ne désarme pas une
  garde en l'enrobant dans un enchaînement.

Le fichier produit est du Python lisible, chez vous. Ouvrez-le, changez
l'ordre, ajoutez une étape — le rechargement à chaud prendra la nouvelle
version. Supprimez-le et la routine disparaît. Et parce qu'une compétence peut
désormais en appeler une autre, la profondeur est bornée et une routine qui se
rappellerait elle-même est arrêtée net, en le disant.


## Architecture

```
IDLE ──« Lily »──▶ WAKE ─▶ LISTEN ─▶ TRANSCRIBE ─▶ THINK
                                                         │
IDLE ◀── suivi expiré ──── SPEAK ◀───── ACT ◀────────────┘
  ▲                          │
  └──── barge-in ────▶ LISTEN
```

Chaque étage est derrière une interface abstraite (`WakeWordEngine`,
`STTEngine`, `LLMEngine`, `TTSEngine`). Changer de moteur est une ligne de
configuration, pas une refonte.

```
lily/
├── plugin.py               ← LE module que les plugins importent
├── app.py                  assemblage config → registre → routeur → pipeline
└── core/
    ├── pipeline.py         machine à états (asyncio)
    ├── memoire.py          historique et faits durables (SQLite)
    ├── apprentissage.py    formulations, corrections, vocabulaire (SQLite)
    ├── semantique.py       index des documents lus, recherche par le sens
    ├── catalogue.py        les signatures de VOS fonctions, lues par AST
    ├── corpus.py           les extraits d'éveil gardés pour réapprendre votre voix
    ├── journal.py          les derniers gestes, pour apprendre une routine
    ├── atelier.py          la frontière entre Lily et vos fichiers
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
        ├── embeddings/     ollama, llamacpp, hachage (repli sans modèle)
        ├── stt/            faster-whisper, vosk, scripted
        ├── tts/            piper, silent
        ├── vad/            silero (onnxruntime), énergie, scripted
        └── wake/           openwakeword, vosk, scripted
tools/entrainer_lily.py entraînement du mot d'éveil, et mesure du seuil
tools/banc_de_code.py       mesure ce qui fait vraiment coder Lily
plugins/                    ← LE dossier
config/                     default.toml, pc.toml, pi.toml, persona.txt
```

### Le routage, en trois étages

Un modèle 7-8B quantifié en Q4, en français, avec une vingtaine d'outils dans
son contexte, hallucine des noms d'outils et se trompe de types d'arguments
assez souvent pour casser une démonstration. Sur Pi avec un 1-3B, c'est pire.
D'où :

1. **Étage déterministe** — score entre la phrase entendue et les `examples` du
   décorateur, **plus vos propres formulations retenues** des tours précédents.
   Latence nulle, aucun modèle sollicité. « Lance un dé à vingt faces » est
   résolu ici, `faces=20` compris.
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
fois par heure ; une Lily qui commente chaque sauvegarde devient vite
insupportable.

### L'écoute permanente

**Un seul thread tient le micro**, ouvert du début à la fin de la session.
C'est ce qui rend le barge-in possible : le micro n'est jamais fermé, même
pendant que Lily parle. Selon le mode courant, chaque trame part vers le
détecteur de mot d'éveil, vers le découpeur d'énoncé, ou vers la surveillance
d'interruption. Le thread ne décide de rien : il émet des événements que la
boucle asyncio consomme.

**Terminer une phrase sans couper l'utilisateur** demande trois précautions :
un silence exigé plus long qu'on ne le croit (700 ms), un **pré-roll** qui
conserve l'audio *précédant* la détection — sans quoi la première syllabe est
perdue — et une durée minimale de parole, pour qu'une porte qui claque ne
devienne pas une commande.

**Le barge-in doit composer avec l'écho.** Sans annulation acoustique, le
micro entend le haut-parleur et Lily se coupe elle-même. Trois garde-fous
réglables : un seuil plus haut que pour l'écoute normale (0,85), un délai de
garde au début de la réponse, et une durée de parole soutenue exigée. Au
casque, on peut tout abaisser. Sur haut-parleur ouvert — le cas d'un Pi —
`barge_in.mode = "eveil"` (n'interrompre que si l'on redit « Lily ») reste
le plus sûr ; c'est le défaut du profil Pi.

**Le mode suivi** garde l'écoute ouverte quelques secondes après la réponse :
on enchaîne sans redire le nom. `assistant.follow_up_seconds = 0` le désactive.

### Configuration

Quatre couches, de la plus faible à la plus forte :

1. `config/default.toml` ;
2. le profil `config/pc.toml` ou `config/pi.toml`, détecté automatiquement ;
3. les variables d'environnement `LILY_SECTION__CLE` ;
4. les options de ligne de commande.

```bash
LILY_LLM__MODEL=qwen2.5:3b-instruct-q4_K_M python main.py --text
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

370 tests, aucun modèle téléchargé, aucun périphérique audio requis. Le critère
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
| 3 | Mot d'éveil « Lily » (`openWakeWord` + repli Vosk), VAD (`silero`), barge-in, mode suivi | **fait** |
| 4 | Rechargement à chaud (`watchdog`), isolation en sous-processus, confirmation, quatre plugins d'exemple | **fait** |
| 5 | Profil Raspberry Pi, mesures de latence, guide d'installation par plateforme | **fait** |

### Entraîner le mot d'éveil

```bash
python tools/entrainer_lily.py preparer      # config + état des prérequis
python tools/entrainer_lily.py echantillons  # positifs français, voix Piper
python tools/entrainer_lily.py entrainer     # pipeline officiel openWakeWord
python tools/entrainer_lily.py installer     # copie le modèle dans models/wake/
python tools/entrainer_lily.py essayer a.wav # vérifie sur un enregistrement
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
*abandonné* — Lily répond et le met en quarantaine — mais son thread
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

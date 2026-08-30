"""Ouvrir ce que l'UTF-8 ne sait pas lire : Word, Excel, PowerPoint, PDF.

L'atelier écrit et lit du texte. Un ``.docx`` ou un ``.xlsx`` n'en est pas :
c'est une archive de XML, et le lire comme du texte ne donne que du charabia.
Ces compétences passent donc par les bibliothèques du format.

**Lecture seulement.** Écrire dans un document Office demande de préserver
styles, formules et mises en page — c'est un autre métier, et le faire à
moitié abîmerait vos fichiers. L'atelier refuse d'ailleurs désormais d'écrire
du texte dans un fichier binaire, plutôt que de le détruire en silence.

Les dépendances sont importées **format par format** : un ``.docx`` reste
lisible même si ``openpyxl`` manque. Une bibliothèque absente donne un message
qui nomme le paquet, jamais une trace d'exception.
"""

import csv as csv_stdlib
from pathlib import Path

from capucine.plugin import (
    SkillRefused,
    atelier,
    demander_au_modele,
    get_config,
    get_logger,
    skill,
)

CONFIG_DEFAULTS = {
    "taille_max_mo": 25,
    "caracteres_max": 20000,
    "lignes_tableur": 15,
    "pages_pdf": 10,
    "documents_fouilles": 40,
}

PAQUETS = {
    ".docx": "python-docx", ".docm": "python-docx",
    ".xlsx": "openpyxl", ".xlsm": "openpyxl",
    ".pptx": "python-pptx",
    ".pdf": "pypdf",
}
EXTENSIONS = frozenset({*PAQUETS, ".csv", ".tsv", ".txt", ".md"})


def _exiger(module: str, extension: str):
    """Importe la bibliothèque du format, ou refuse en nommant le paquet."""
    try:
        return __import__(module)
    except ImportError:
        paquet = PAQUETS.get(extension, module)
        raise SkillRefused(
            f"Je ne peux pas ouvrir un fichier « {extension} » : le paquet "
            f"« {paquet} » n'est pas installé. Ajoutez-le avec : pip install {paquet}"
        ) from None


def _ouvrir(chemin: str) -> Path:
    cible = atelier().resoudre(chemin, doit_exister=True)
    if cible.is_dir():
        raise SkillRefused(f"« {cible.name} » est un dossier, pas un document.")
    taille_mo = cible.stat().st_size / 1_000_000
    maximum = float(get_config("taille_max_mo", 25))
    if taille_mo > maximum:
        raise SkillRefused(
            f"« {cible.name} » fait {taille_mo:.0f} méga-octets, au-delà de la limite "
            f"de {maximum:.0f}. Relevez documents.taille_max_mo si c'est voulu."
        )
    return cible


# --- extraction, format par format ------------------------------------------

def _texte_docx(chemin: Path) -> str:
    docx = _exiger("docx", chemin.suffix.lower())
    document = docx.Document(str(chemin))
    morceaux = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    # Les tableaux d'un Word portent souvent l'essentiel : on ne les saute pas.
    for tableau in document.tables:
        for ligne in tableau.rows:
            cellules = [cellule.text.strip() for cellule in ligne.cells]
            if any(cellules):
                morceaux.append(" | ".join(cellules))
    return "\n".join(morceaux)


def _texte_xlsx(chemin: Path, feuille: str = "", lignes: int = 0) -> str:
    openpyxl = _exiger("openpyxl", chemin.suffix.lower())
    # data_only : on veut le résultat des formules, pas « =SOMME(A1:A9) ».
    classeur = openpyxl.load_workbook(str(chemin), read_only=True, data_only=True)
    try:
        if feuille:
            correspondance = next(
                (nom for nom in classeur.sheetnames if feuille.lower() in nom.lower()), None
            )
            if correspondance is None:
                raise SkillRefused(
                    f"Pas de feuille « {feuille} ». Il y a : {', '.join(classeur.sheetnames)}."
                )
            feuilles = [classeur[correspondance]]
        else:
            feuilles = list(classeur.worksheets)

        maximum = lignes or int(get_config("lignes_tableur", 15))
        morceaux: list[str] = []
        for onglet in feuilles:
            morceaux.append(f"— feuille « {onglet.title} » —")
            for numero, ligne in enumerate(onglet.iter_rows(values_only=True), 1):
                if numero > maximum:
                    morceaux.append("…")
                    break
                cellules = ["" if v is None else str(v) for v in ligne]
                if any(cellules):
                    morceaux.append(" | ".join(cellules))
        return "\n".join(morceaux)
    finally:
        classeur.close()


def _texte_pptx(chemin: Path) -> str:
    pptx = _exiger("pptx", chemin.suffix.lower())
    presentation = pptx.Presentation(str(chemin))
    morceaux: list[str] = []
    for numero, diapositive in enumerate(presentation.slides, 1):
        morceaux.append(f"— diapositive {numero} —")
        for forme in diapositive.shapes:
            if getattr(forme, "has_text_frame", False):
                texte = forme.text_frame.text.strip()
                if texte:
                    morceaux.append(texte)
        # Les notes du présentateur disent souvent ce que la diapositive tait.
        if diapositive.has_notes_slide:
            notes = diapositive.notes_slide.notes_text_frame.text.strip()
            if notes:
                morceaux.append(f"(notes) {notes}")
    return "\n".join(morceaux)


def _texte_pdf(chemin: Path, pages: int = 0) -> str:
    pypdf = _exiger("pypdf", chemin.suffix.lower())
    lecteur = pypdf.PdfReader(str(chemin))
    if lecteur.is_encrypted:
        raise SkillRefused(f"« {chemin.name} » est protégé par un mot de passe.")
    maximum = pages or int(get_config("pages_pdf", 10))
    morceaux = []
    for numero, page in enumerate(lecteur.pages[:maximum], 1):
        texte = (page.extract_text() or "").strip()
        if texte:
            morceaux.append(f"— page {numero} —\n{texte}")
    if len(lecteur.pages) > maximum:
        morceaux.append(f"… et {len(lecteur.pages) - maximum} pages de plus")
    return "\n".join(morceaux)


def _texte_csv(chemin: Path, lignes: int = 0) -> str:
    maximum = lignes or int(get_config("lignes_tableur", 15))
    separateur = "\t" if chemin.suffix.lower() == ".tsv" else ","
    with chemin.open(newline="", encoding="utf-8", errors="replace") as fichier:
        lecteur = csv_stdlib.reader(fichier, delimiter=separateur)
        morceaux = [" | ".join(ligne) for numero, ligne in enumerate(lecteur) if numero < maximum]
    return "\n".join(morceaux)


def extraire(chemin: Path, lignes: int = 0) -> str:
    """Le texte d'un document, quel que soit son format."""
    extension = chemin.suffix.lower()
    if extension in (".docx", ".docm"):
        return _texte_docx(chemin)
    if extension in (".xlsx", ".xlsm"):
        return _texte_xlsx(chemin, lignes=lignes)
    if extension == ".pptx":
        return _texte_pptx(chemin)
    if extension == ".pdf":
        return _texte_pdf(chemin, pages=lignes)
    if extension in (".csv", ".tsv"):
        return _texte_csv(chemin, lignes=lignes)
    if extension in (".txt", ".md"):
        return chemin.read_text(encoding="utf-8", errors="replace")
    if extension in (".doc", ".xls", ".ppt"):
        raise SkillRefused(
            f"« {extension} » est l'ancien format binaire de Microsoft, que je ne sais "
            "pas lire. Réenregistrez le fichier en .docx, .xlsx ou .pptx."
        )
    raise SkillRefused(
        f"Je ne sais pas ouvrir un « {extension} ». Je lis le Word, l'Excel, le "
        "PowerPoint, le PDF et le CSV."
    )


# --- compétences ------------------------------------------------------------

@skill(
    description="Ouvre un document Word, Excel, PowerPoint, PDF ou CSV et en lit le contenu.",
    examples=[
        "ouvre le document word",
        "lis le fichier excel du budget",
        "montre-moi la présentation powerpoint",
        "ouvre ce pdf",
    ],
    timeout=60.0,
)
def lire_document(chemin: str, lignes: int = 0) -> dict:
    """Extrait le texte d'un document, quel que soit son format.

    Args:
        chemin: Le document à ouvrir, relatif à l'atelier.
        lignes: Nombre de lignes ou de pages à lire. Zéro prend la valeur configurée.
    """
    cible = _ouvrir(chemin)
    texte = extraire(cible, lignes=int(lignes))
    if not texte.strip():
        return {
            "speak": f"{cible.name} ne contient pas de texte que je sache lire.",
            "display": f"{cible.name} : extraction vide "
                       "(document scanné ou uniquement des images ?)",
        }
    maximum = int(get_config("caracteres_max", 20000))
    get_logger().info("Document lu : %s (%d caractères)", cible.name, len(texte))
    return {
        "speak": texte[:600],
        "display": texte[:maximum] + ("\n…" if len(texte) > maximum else ""),
    }


@skill(
    description="Résume un document Word, Excel, PowerPoint ou PDF.",
    examples=["résume ce document", "de quoi parle ce rapport", "fais-moi un résumé du pdf"],
    timeout=180.0,
)
def resumer_document(chemin: str) -> dict:
    """Extrait le document puis le fait résumer par le modèle local.

    Args:
        chemin: Le document à résumer, relatif à l'atelier.
    """
    cible = _ouvrir(chemin)
    texte = extraire(cible)
    if not texte.strip():
        raise SkillRefused(f"{cible.name} ne contient aucun texte extractible.")

    extrait = texte[: int(get_config("caracteres_max", 20000))]
    resume = demander_au_modele(
        f"Résume ce document en trois phrases, en français :\n\n{extrait}",
        system="Tu résumes brièvement et précisément. Pas de liste, pas de titre, "
               "pas de formule d'introduction.",
        max_tokens=350,
    ).strip()
    return {
        "speak": resume or "Je n'arrive pas à le résumer.",
        "display": f"{cible.name} ({len(texte)} caractères)\n\n{resume}",
    }


@skill(
    description="Dit quelles feuilles contient un classeur Excel, et leur taille.",
    examples=["quelles feuilles il y a dans le classeur", "montre-moi les onglets du fichier excel"],
    timeout=60.0,
)
def feuilles_du_tableur(chemin: str) -> dict:
    """Liste les onglets d'un fichier Excel avec leurs dimensions.

    Args:
        chemin: Le classeur, relatif à l'atelier.
    """
    cible = _ouvrir(chemin)
    if cible.suffix.lower() not in (".xlsx", ".xlsm"):
        raise SkillRefused(f"« {cible.name} » n'est pas un classeur Excel.")

    openpyxl = _exiger("openpyxl", cible.suffix.lower())
    classeur = openpyxl.load_workbook(str(cible), read_only=True, data_only=True)
    try:
        details = [
            (onglet.title, onglet.max_row or 0, onglet.max_column or 0)
            for onglet in classeur.worksheets
        ]
    finally:
        classeur.close()

    noms = ", ".join(nom for nom, _, _ in details)
    return {
        "speak": f"{len(details)} feuille{'s' if len(details) > 1 else ''} : {noms}.",
        "display": "\n".join(
            f"{nom} — {lignes} ligne(s) × {colonnes} colonne(s)"
            for nom, lignes, colonnes in details
        ),
    }


@skill(
    description="Lit une feuille précise d'un classeur Excel.",
    examples=[
        "lis la feuille synthèse du classeur",
        "montre-moi l'onglet résultats",
        "les vingt premières lignes du tableur",
    ],
    timeout=60.0,
)
def lire_tableur(chemin: str, feuille: str = "", lignes: int = 0) -> dict:
    """Lit les premières lignes d'une feuille de calcul.

    Les formules sont rendues par leur **résultat** et non par leur écriture :
    « 42 » est plus utile à entendre que « =SOMME(A1:A9) ». Attention toutefois,
    c'est le résultat *mis en cache par Excel* qui est lu — une formule qu'Excel
    n'a jamais calculée (fichier produit par un script, par exemple) apparaît
    vide. Rien ne calcule les formules à sa place.

    Args:
        chemin: Le classeur, relatif à l'atelier.
        feuille: Le nom de l'onglet. Vide, toutes les feuilles.
        lignes: Combien de lignes lire. Zéro prend la valeur configurée.
    """
    cible = _ouvrir(chemin)
    if cible.suffix.lower() in (".csv", ".tsv"):
        texte = _texte_csv(cible, lignes=int(lignes))
    else:
        if cible.suffix.lower() not in (".xlsx", ".xlsm"):
            raise SkillRefused(f"« {cible.name} » n'est pas un classeur.")
        texte = _texte_xlsx(cible, feuille=feuille, lignes=int(lignes))

    premieres = [ligne for ligne in texte.splitlines() if not ligne.startswith("—")][:3]
    return {
        "speak": " ; ".join(premieres)[:400] or "Cette feuille est vide.",
        "display": texte[: int(get_config("caracteres_max", 20000))],
    }


@skill(
    description="Cherche un mot dans les documents Word, Excel, PowerPoint et PDF du projet.",
    examples=[
        "cherche le mot budget dans les documents",
        "quel document parle de la valorisation",
        "trouve ce terme dans mes fichiers word",
    ],
    timeout=300.0,
)
def chercher_dans_documents(texte: str, motif: str = "*") -> dict:
    """Fouille les documents de l'atelier.

    Args:
        texte: Ce qu'il faut trouver.
        motif: Filtre sur les noms de fichiers, par exemple « *.docx ».
    """
    cible = texte.strip().lower()
    if not cible:
        raise SkillRefused("Que dois-je chercher ?")

    espace = atelier()
    maximum = int(get_config("documents_fouilles", 40))
    examines = 0
    trouvailles: list[str] = []
    ignores: list[str] = []

    for racine in espace.racines:
        for fichier in sorted(racine.rglob(motif)):
            if examines >= maximum:
                break
            if not fichier.is_file() or fichier.suffix.lower() not in EXTENSIONS:
                continue
            if any(partie.startswith(".") for partie in fichier.parts):
                continue
            try:
                espace._verifier_interdits(fichier)
                contenu = extraire(fichier)
            except SkillRefused as exc:
                ignores.append(f"{fichier.name} : {exc}")
                continue
            except Exception as exc:
                ignores.append(f"{fichier.name} : illisible ({type(exc).__name__})")
                continue
            examines += 1
            if cible in contenu.lower():
                ligne = next(
                    (texte_ligne.strip() for texte_ligne in contenu.splitlines()
                     if cible in texte_ligne.lower()),
                    "",
                )
                trouvailles.append(f"{fichier.name} : {ligne[:120]}")

    if not trouvailles:
        journal = f"0 résultat sur {examines} document(s) examiné(s)"
        if ignores:
            journal += "\n" + "\n".join(f"ignoré — {raison}" for raison in ignores[:5])
        return {"speak": f"Rien sur « {texte} » dans vos documents.", "display": journal}

    noms = [trouvaille.split(" : ")[0] for trouvaille in trouvailles]
    return {
        "speak": (
            f"{len(trouvailles)} document{'s' if len(trouvailles) > 1 else ''} : "
            + ", ".join(noms[:4]) + "."
        ),
        "display": "\n".join(trouvailles),
    }

"""
Jour 3 - Feature Engineering.

    data/processed/movies_clean.csv  ->  data/features/movies_features.csv
                                     ->  data/features/feature_groups.json
                                     ->  reports/feature_justifications.md

Usage :
    python src/features/build_features.py

PRINCIPE ANTI-LEAKAGE
---------------------
1. Toutes les features créées ici sont calculées LIGNE PAR LIGNE (année, nombre de genres,
   log du budget...). Aucune ne dépend de statistiques globales du jeu de données
   (moyenne, quantiles, encodage par la cible) : pas de fuite entre train et test.
   -> C'est pourquoi les catégories de durée utilisent des seuils FIXES (90/120/150 min)
      et non des quantiles.
2. Les variables qui mesurent l'engagement du public (vote_count, vote_average, popularity)
   et le succès commercial (revenue) sont CONSERVÉES dans le fichier (utiles pour l'EDA et
   pour créer la cible) mais listées dans "leakage_risk" : elles ne seront pas des features.
3. Aucune imputation ici : elle sera faite dans un Pipeline, ajustée sur le train seul.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.cleaning.clean_data import CLEAN_CSV, LIST_COLS, LIST_SEP, load_clean  # noqa: E402

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_CSV = FEATURES_DIR / "movies_features.csv"
GROUPS_JSON = FEATURES_DIR / "feature_groups.json"
JUSTIF_MD = PROJECT_ROOT / "reports" / "feature_justifications.md"

RUNTIME_BINS = [0, 90, 120, 150, np.inf]          # seuils fixes (pas de quantiles)
RUNTIME_LABELS = ["court", "moyen", "long", "tres_long"]

logger = logging.getLogger("features")

# nom de la feature -> (groupe, justification)
FEATURE_JUSTIFICATIONS = {
    "release_year": ("numeric", "Époque de sortie : les goûts et le volume de films varient avec les années."),
    "month_sin": ("numeric", "Mois encodé de façon cyclique (décembre est proche de janvier)."),
    "month_cos": ("numeric", "Seconde composante du mois cyclique (sin et cos sont nécessaires ensemble)."),
    "n_genres": ("numeric", "Un film multi-genres touche un public plus large."),
    "n_keywords": ("numeric", "Richesse de l'indexation : indicateur de la notoriété du film sur TMDB."),
    "runtime": ("numeric", "Durée brute en minutes."),
    "budget_log": ("numeric", "log(1+budget) : le budget est très asymétrique, le log rend la relation exploitable."),
    "overview_word_count": ("numeric", "Longueur du synopsis (mots) : les films documentés ont souvent un synopsis plus détaillé."),
    "overview_char_len": ("numeric", "Longueur du synopsis (caractères)."),
    "title_word_count": ("numeric", "Longueur du titre en mots."),
    "has_budget": ("binary", "Budget connu ou non : l'absence d'information est elle-même un signal (les petits films sont moins documentés)."),
    "is_english": ("binary", "Langue originale anglaise : marché et audience potentielle beaucoup plus larges."),
    "has_overview": ("binary", "Synopsis présent."),
    "genre_*": ("binary", "One-hot multi-label des genres (un film peut avoir plusieurs genres). Vocabulaire fermé TMDB : aucune fuite."),
    "runtime_category": ("categorical", "Court / moyen / long / très long (seuils fixes 90, 120, 150 min)."),
    "original_language": ("categorical", "Langue originale (regroupement des langues rares fait dans le Pipeline)."),
    "release_month": ("descriptive_only", "Mois de sortie (EDA). Le modèle utilise sa version cyclique sin/cos."),
    "release_quarter": ("descriptive_only", "Trimestre de sortie (EDA, saisonnalité)."),
    "release_decade": ("descriptive_only", "Décennie (EDA, graphiques par période). Redondante avec release_year."),
}

LEAKAGE_RISK = {
    "vote_count": "Sert à construire la cible high_engagement : interdit comme feature.",
    "vote_average": "Résultat direct de l'engagement du public (les votes) : proche de la cible.",
    "popularity": "Score TMDB calculé à partir des vues, votes et favoris : proxy de l'engagement.",
    "revenue": "Conséquence du succès du film, connue seulement après la sortie.",
}


def slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables dérivées. Aucune statistique globale n'est utilisée."""
    out = df.copy()

    # --- Dates -------------------------------------------------------------
    d = out["release_date"]
    out["release_year"] = d.dt.year
    out["release_month"] = d.dt.month
    out["release_quarter"] = d.dt.quarter
    out["release_decade"] = (out["release_year"] // 10) * 10
    out["month_sin"] = np.sin(2 * np.pi * out["release_month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["release_month"] / 12)

    # --- Comptages ---------------------------------------------------------
    out["n_genres"] = out["genres"].apply(len)
    out["n_keywords"] = out["keywords"].apply(len)

    # --- Durée -------------------------------------------------------------
    out["runtime_category"] = pd.cut(
        out["runtime"], bins=RUNTIME_BINS, labels=RUNTIME_LABELS, right=False
    ).astype(object)

    # --- Budget ------------------------------------------------------------
    out["has_budget"] = out["budget"].notna().astype(int)
    out["budget_log"] = np.log1p(out["budget"])

    # --- Langue ------------------------------------------------------------
    out["is_english"] = (out["original_language"] == "en").astype(int)

    # --- Texte (longueurs uniquement : le contenu sera traité par TF-IDF) ---
    overview = out["overview"].fillna("")
    out["has_overview"] = (overview.str.strip() != "").astype(int)
    out["overview_word_count"] = overview.str.split().str.len()
    out["overview_char_len"] = overview.str.len()
    out["title_word_count"] = out["title"].astype(str).str.split().str.len()

    # --- Genres : one-hot multi-label --------------------------------------
    for g in sorted({g for lst in out["genres"] for g in lst}):
        out[f"genre_{slugify(g)}"] = out["genres"].apply(lambda lst, g=g: int(g in lst))

    return out


def feature_groups(df: pd.DataFrame) -> dict:
    """Regroupe les colonnes pour le ColumnTransformer (Jour 5)."""
    genre_cols = sorted(c for c in df.columns if c.startswith("genre_"))
    groups = {"numeric": [], "binary": [], "categorical": [], "descriptive_only": []}
    for name, (grp, _) in FEATURE_JUSTIFICATIONS.items():
        if name == "genre_*":
            groups["binary"] += genre_cols
        elif name in df.columns:
            groups[grp].append(name)
    groups["text"] = ["overview"]
    groups["leakage_risk"] = list(LEAKAGE_RISK)
    groups["identifiers"] = ["movie_id", "title"]
    return groups


def write_justifications(path: Path = JUSTIF_MD) -> None:
    lines = ["# Justification des variables créées", "",
             "| Variable | Groupe | Justification |", "|---|---|---|"]
    lines += [f"| `{n}` | {g} | {j} |" for n, (g, j) in FEATURE_JUSTIFICATIONS.items()]
    lines += ["", "## Variables écartées des features (risque de Data Leakage)", "",
              "| Variable | Raison |", "|---|---|"]
    lines += [f"| `{n}` | {r} |" for n, r in LEAKAGE_RISK.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_features(df: pd.DataFrame, path: Path = FEATURES_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in LIST_COLS:
        out[c] = out[c].apply(lambda lst: LIST_SEP.join(lst))
    out.to_csv(path, index=False, encoding="utf-8")


def load_features(path: Path = FEATURES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["release_date"])
    for c in LIST_COLS:
        df[c] = df[c].fillna("").apply(lambda s: [x for x in s.split(LIST_SEP) if x])
    return df


def run_feature_engineering() -> pd.DataFrame:
    df = load_clean(CLEAN_CSV)
    feats = build_features(df)
    groups = feature_groups(feats)
    save_features(feats)
    GROUPS_JSON.write_text(json.dumps(groups, indent=2, ensure_ascii=False), encoding="utf-8")
    write_justifications()

    print(f"Films : {len(feats)} | colonnes : {df.shape[1]} -> {feats.shape[1]}")
    for k, v in groups.items():
        shown = v if len(v) <= 8 else v[:8] + [f"... (+{len(v) - 8})"]
        print(f"  {k:17s} ({len(v):2d}) : {shown}")
    print(f"\nFichier : {FEATURES_CSV}\nGroupes : {GROUPS_JSON}\nRapport : {JUSTIF_MD}")
    return feats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_feature_engineering()

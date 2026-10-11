import io
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_MOVIES_DIR = PROJECT_ROOT / "data" / "raw" / "movies"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
CLEAN_CSV = PROCESSED_DIR / "movies_clean.csv"
COLUMN_TYPES_JSON = PROCESSED_DIR / "column_types.json"
REPORT_TXT = REPORTS_DIR / "cleaning_report.txt"

LIST_SEP = "|" 
MIN_MONEY = 1_000  # budget/revenue en dessous 
MAX_RUNTIME = 300  # 300 minutes est la limite (signalé, non supprimé)

# Typologie des variables 
ID_COLS = ["movie_id"]
NUMERIC_COLS = ["runtime", "budget", "revenue", "popularity", "vote_average", "vote_count"]
CATEGORICAL_COLS = ["original_language", "status"]
TEXT_COLS = ["title", "overview"]
DATE_COLS = ["release_date"]
LIST_COLS = ["genres", "keywords"]  

logger = logging.getLogger("cleaning")


# 1.Lit tous les JSON bruts en df 

def load_raw_movies(raw_dir: Path = RAW_MOVIES_DIR) -> pd.DataFrame:
    rows = []

    for f in sorted(Path(raw_dir).glob("*.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Fichier illisible %s : %s", f.name, exc)
            continue

        keywords = (m.get("keywords") or {}).get("keywords") or []
        rows.append(
            {
                "movie_id": m.get("id"),
                "title": m.get("title"),
                "overview": m.get("overview"),
                "release_date": m.get("release_date"),
                "runtime": m.get("runtime"),
                "original_language": m.get("original_language"),
                "status": m.get("status"),
                "genres": [g["name"] for g in (m.get("genres") or []) if g.get("name")],
                "keywords": [k["name"] for k in keywords if k.get("name")],
                "budget": m.get("budget"),
                "revenue": m.get("revenue"),
                "popularity": m.get("popularity"),
                "vote_average": m.get("vote_average"),
                "vote_count": m.get("vote_count"),
            }
        )
    logger.info("%d fichiers JSON chargés", len(rows))
    return pd.DataFrame(rows)


# 2. Analyse : types, structure, manquants, doublons, incohérences
#NaN, liste vide et chaine," ",  vide comptés comme manquants 

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = {}
    for col in df.columns:
        s = df[col] 

        if col in LIST_COLS: # []
            miss[col] = int((s.apply(lambda x: len(x) if isinstance(x, list) else 0) == 0).sum())
        elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            miss[col] = int(s.isna().sum())
        else:  
            miss[col] = int((s.isna() | (s.astype(str).str.strip() == "")).sum())
    out = pd.DataFrame({"manquants": pd.Series(miss)})
    out["pourcentage"] = (out["manquants"] / max(len(df), 1) * 100).round(1)  # pour eviter 0
    return out

# Compte les incohérences AVANT correction

def detect_inconsistencies(df: pd.DataFrame) -> dict:

    d = df.copy()
    dates = pd.to_datetime(d["release_date"], errors="coerce") # trsf a Not a Time
    num = {c: pd.to_numeric(d[c], errors="coerce") for c in NUMERIC_COLS}
    today = pd.Timestamp.today().normalize() #garder juste le date 
    return {
        "movie_id dupliqués": int(d["movie_id"].duplicated().sum()),
        "(title, release_date) dupliqués": int(d.duplicated(subset=["title", "release_date"]).sum()),
        "budget = 0 (valeur bidon)": int((num["budget"] == 0).sum()),
        f"budget entre 1 et {MIN_MONEY - 1} (erreur de saisie)": int(num["budget"].between(1, MIN_MONEY - 1).sum()),
        "revenue = 0 (valeur bidon)": int((num["revenue"] == 0).sum()),
        f"revenue entre 1 et {MIN_MONEY - 1} (erreur de saisie)": int(num["revenue"].between(1, MIN_MONEY - 1).sum()),
        "runtime = 0 ou négatif": int((num["runtime"] <= 0).sum()),
        f"runtime > {MAX_RUNTIME} min (suspect)": int((num["runtime"] > MAX_RUNTIME).sum()),
        "vote_count = 0 mais vote_average > 0": int(((num["vote_count"] == 0) & (num["vote_average"] > 0)).sum()),
        "vote_average hors [0, 10]": int((~num["vote_average"].between(0, 10) & num["vote_average"].notna()).sum()),
        "popularity négative": int((num["popularity"] < 0).sum()),
        "date illisible": int((dates.isna() & d["release_date"].notna() & (d["release_date"] != "")).sum()),
        "date dans le futur": int((dates > today).sum()),
        "status différent de 'Released'": int((d["status"].notna() & (d["status"] != "Released")).sum()),
    }

# Rapport texte 

def analyze(df: pd.DataFrame, title: str) -> str:
    buf = io.StringIO()
    df.info(buf=buf)   # ecrris dans buf
    lines = [
        f"=== {title} ===",
        f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes",
        "",
        "--- Types et structure (df.info) ---",
        buf.getvalue(), 
        "--- Valeurs manquantes ---",
        missing_summary(df).to_string(),
        "",
    ]
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    if num_cols:
        num = df[num_cols].apply(pd.to_numeric, errors="coerce")
        lines += ["--- Statistiques numériques ---", num.describe().round(2).T.to_string(), ""]
    return "\n".join(lines)


# 3. Nettoyage

def clean_movies(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:

    log = {}
    df = df.copy()
    n0 = len(df)

    # a - sans id ou sans titre
    df["title"] = df["title"].astype("string").str.strip()
    df = df[df["movie_id"].notna() & df["title"].notna() & (df["title"] != "")]
    log["lignes sans id/titre supprimées"] = n0 - len(df)

    # b) Doublons 
    n = len(df)
    df = df.drop_duplicates(subset="movie_id", keep="first")
    log["doublons movie_id supprimés"] = n - len(df)

    n = len(df)
    df = df.sort_values("vote_count", ascending=False)  
    df = df.drop_duplicates(subset=["title", "release_date"], keep="first")
    log["doublons (title, release_date) supprimés"] = n - len(df)

    # c) Conversion des types
    df["movie_id"] = df["movie_id"].astype(int)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["title"] = df["title"].astype(object)

    # d) Valeurs bidon -> NaN 

    for c in ("budget", "revenue"):
        mask = df[c] < MIN_MONEY # t f 
        log[f"{c} mis à NaN (< {MIN_MONEY})"] = int(mask.sum())
        df.loc[mask, c] = np.nan
    mask = df["runtime"] <= 0
    log["runtime mis à NaN (<= 0)"] = int(mask.sum())
    df.loc[mask, "runtime"] = np.nan

    # e) Note sans vote : la moyenne n'a aucun sens
    mask = (df["vote_count"] == 0) & (df["vote_average"] == 0)
    log["vote_average mis à NaN (0 vote)"] = int(mask.sum())
    df.loc[mask, "vote_average"] = np.nan

    # f) Texte : espaces superflus, texte vide -> NaN
    df["overview"] = df["overview"].astype("string").str.strip().replace("", pd.NA).astype(object)
    df["overview"] = df["overview"].where(df["overview"].notna(), np.nan)

    # g) Catégorielles : normalisation
    df["original_language"] = df["original_language"].astype("string").str.strip().str.lower().astype(object)
    df["status"] = df["status"].astype("string").str.strip().astype(object)

    # h) Films non sortis 
    n = len(df)
    df = df[df["status"].isna() | (df["status"] == "Released")]
    log["films non 'Released' supprimés"] = n - len(df)

    # i) Listes : garantir des listes (jamais NaN)
    for c in LIST_COLS:
        df[c] = df[c].apply(lambda x: x if isinstance(x, list) else [])

    df = df.sort_values("movie_id").reset_index(drop=True)
    log["lignes finales"] = len(df)
    return df, log

# Sépare les colonnes en numériques / catégorielles / textuelles / dates / listes
 
def split_variable_types(df: pd.DataFrame) -> dict:
    groups = {
        "id": ID_COLS,
        "numeric": NUMERIC_COLS,
        "categorical": CATEGORICAL_COLS,
        "text": TEXT_COLS,
        "date": DATE_COLS,
        "list": LIST_COLS,
    }
    return {k: [c for c in v if c in df.columns] for k, v in groups.items()}

# 4. Sauvegarde / rechargement

def save_clean(df: pd.DataFrame, path: Path = CLEAN_CSV) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in LIST_COLS:
        out[c] = out[c].apply(lambda lst: LIST_SEP.join(x.replace(LIST_SEP, " ") for x in lst))
    out.to_csv(path, index=False, encoding="utf-8")
    COLUMN_TYPES_JSON.write_text(
        json.dumps(split_variable_types(df), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_clean(path: Path = CLEAN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["release_date"])
    for c in LIST_COLS:
        df[c] = df[c].fillna("").apply(lambda s: [x for x in s.split(LIST_SEP) if x])
    return df


# 5. Orchestration

def run_cleaning() -> pd.DataFrame:
    raw = load_raw_movies()
    if raw.empty:
        raise SystemExit("Aucun JSON dans data/raw/movies : lancer l'extraction (Jour 1)")

    report = [analyze(raw, "AVANT NETTOYAGE")]
    report += ["=== INCOHÉRENCES DÉTECTÉES (avant correction) ==="]
    report += [f"{k:55s}: {v}" for k, v in detect_inconsistencies(raw).items()] + [""]

    clean, log = clean_movies(raw)
    report += ["=== JOURNAL DU NETTOYAGE ==="] + [f"{k:55s}: {v}" for k, v in log.items()] + [""]
    report.append(analyze(clean, "APRÈS NETTOYAGE"))
    report += ["=== TYPOLOGIE DES VARIABLES ==="]
    report += [f"{k:12s}: {v}" for k, v in split_variable_types(clean).items()]

    text = "\n".join(report)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text(text, encoding="utf-8")
    save_clean(clean)
    print(text)
    print(f"\nFichier propre : {CLEAN_CSV}\nRapport        : {REPORT_TXT}")
    return clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_cleaning()

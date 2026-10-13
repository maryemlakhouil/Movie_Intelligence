"""
tfidf = TfidfVectorizer(
                       stop_words = "english",  #supprimer les (is,the,and,...)
                       max_features = 5000 # khali 5000 darorin 
                       min_df  = 2 (1) # ignorer les mots rare 
                       max_df = 0.8(1.0) #ignorer les mots trop frequents
                    sublinear_tf = true # deminuer imapct de mot fi halat repetitions 
)
"""
import argparse
import logging
import re
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features.build_features import FEATURES_DIR, load_features  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = FEATURES_DIR / "tfidf_matrix.npz"
IDS_PATH = FEATURES_DIR / "tfidf_movie_ids.csv"
TEXT_CSV = FEATURES_DIR / "movies_text.csv"
EXPERIMENTS_CSV = REPORTS_DIR / "tfidf_experiments.csv"
REPORT_MD = REPORTS_DIR / "tfidf_report.md"

MAX_FEATURES_GRID = [500, 1000, 2000, 5000, None]
NGRAM_GRID = [(1, 1), (1, 2), (1, 3)]
DEFAULT_MAX_FEATURES, DEFAULT_NGRAM = 5000, (1, 2)
MIN_DF, MAX_DF = 2, 0.8

DOMAIN_STOPWORDS = {
    "film", "movie", "story", "dont", "doesnt", "didnt", "isnt", "wasnt", "cant", "couldnt",
    "wouldnt", "shouldnt", "arent", "werent", "hasnt", "havent", "hadnt", "im", "ive",
    "youre", "theyre", "thats", "theres",
}

# Métadonnées TMDB et noms de genres : redondants avec `genres`, et rendraient l'évaluation circulaire
DOMAIN_STOPWORDS |= {"duringcreditsstinger", "aftercreditsstinger"}
DOMAIN_STOPWORDS |= {"action", "adventure", "animation", "comedy", "crime", "documentary", "drama", "fantasy",
                     "horror", "mystery", "romance", "thriller", "western", "science", "fiction"}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS)

logger = logging.getLogger("tfidf")
sns.set_theme(style="whitegrid", context="notebook")


# 1. Nettoyage du texte 

def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"<[^>]+>", " ", t)                   
    t = re.sub(r"http\S+|www\.\S+", " ", t)           
    t = re.sub(r"[’']s\b", "", t)                    
    t = re.sub(r"[’']", "", t)                       
    t = re.sub(r"[^a-zà-ÿ\s]", " ", t)                
    return re.sub(r"\s+", " ", t).strip()

# Valeurs manquantes -> chaîne vide 
def build_text(df: pd.DataFrame, include_keywords: bool = False) -> pd.Series:
    text = df["overview"].fillna("").astype(str)
    if include_keywords:
        text = text + " " + df["keywords"].apply(lambda k: " ".join(k))
    return text


def clean_series(texts: pd.Series) -> pd.Series:
    return texts.map(clean_text)


# 2. Vectoriseur

def make_vectorizer(max_features=DEFAULT_MAX_FEATURES, ngram_range=DEFAULT_NGRAM,
                    min_df=MIN_DF, max_df=MAX_DF) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words=STOP_WORDS,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,         
        max_df=max_df,      
        lowercase=False,      # déjà fait 
        norm="l2",         
    )

 
def matrix_stats(X: sp.csr_matrix) -> dict:
    n, v = X.shape
    return {
        "n_docs": n,
        "vocab_size": v,
        "nnz": int(X.nnz),
        "sparsity_pct": round((1 - X.nnz / (n * v)) * 100, 2) if n * v else 0.0,
        "nnz_per_doc": round(X.nnz / max(n, 1), 1),
        "size_mb": round((X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6, 2),
    }


# 3. 1.0 : les deux films les mêmes , 0.0 : aucun genre en commun. Entre 0.0 et 1.0 : ils partagent une parti

def _jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def genre_coherence(X, genres: list, k: int = 5) -> float:
   
    sets = [set(g) for g in genres]
    sim = linear_kernel(X)                      
    np.fill_diagonal(sim, -1.0)                 
    has_vec = np.asarray(X.getnnz(axis=1) > 0).ravel()
    k = min(k, X.shape[0] - 1)
    top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    scores = [
        _jaccard(sets[i], sets[j])
        for i in range(X.shape[0]) if has_vec[i] and sets[i]
        for j in top[i]
    ]
    return float(np.mean(scores)) if scores else 0.0

def random_baseline(genres: list, n_pairs: int = 20000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    sets = [set(g) for g in genres]
    i = rng.integers(0, len(sets), n_pairs)
    j = rng.integers(0, len(sets), n_pairs)
    return float(np.mean([_jaccard(sets[a], sets[b]) for a, b in zip(i, j) if a != b]))

# 4. Expériences max_features x ngram_range

def run_experiments(clean_texts, genres, max_features_grid=MAX_FEATURES_GRID,
                    ngram_grid=NGRAM_GRID, min_df=MIN_DF, max_df=MAX_DF) -> pd.DataFrame:
    rows = []
    for ng in ngram_grid:
        for mf in max_features_grid:
            t0 = time.time()
            X = make_vectorizer(mf, ng, min_df, max_df).fit_transform(clean_texts)
            fit_s = time.time() - t0
            st = matrix_stats(X)
            rows.append({
                "ngram_min": ng[0], "ngram_max": ng[1], "max_features": mf,
                "vocab_size": st["vocab_size"], "nnz_per_doc": st["nnz_per_doc"],
                "sparsity_pct": st["sparsity_pct"], "size_mb": st["size_mb"],
                "genre_coherence": round(genre_coherence(X, genres), 4),
                "fit_seconds": round(fit_s, 2),
            })
            logger.info("max_features=%s ngram=%s -> %d termes, cohérence %.4f",
                        mf, ng, st["vocab_size"], rows[-1]["genre_coherence"])
    df = pd.DataFrame(rows)
    df["max_features"] = df["max_features"].astype("Int64")  
    return df


def select_best(exp: pd.DataFrame, tolerance: float = 0.005):
    ok = exp[exp["genre_coherence"] >= exp["genre_coherence"].max() - tolerance]
    row = ok.sort_values(["vocab_size", "ngram_max"]).iloc[0]
    mf = None if pd.isna(row["max_features"]) else int(row["max_features"])
    return mf, (int(row["ngram_min"]), int(row["ngram_max"]))


# 5. Termes représentatifs

def top_terms_global(vec, X, n: int = 20) -> pd.Series:
    means = np.asarray(X.mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    idx = means.argsort()[::-1][:n]
    return pd.Series(means[idx], index=terms[idx], name="tfidf_moyen")


def top_terms_by_genre(vec, X, genres: list, genre_names: list, n: int = 8, min_docs: int = 5) -> dict:
    terms = vec.get_feature_names_out()
    glob = np.asarray(X.mean(axis=0)).ravel()
    out = {}
    for g in genre_names:
        mask = np.array([g in lst for lst in genres])
        if mask.sum() < min_docs: 
            continue
        diff = np.asarray(X[mask].mean(axis=0)).ravel() - glob
        out[g] = list(terms[diff.argsort()[::-1][:n]])
    return out


def top_terms_for_doc(vec, X, idx: int, n: int = 8) -> pd.Series:
    row = X[idx].toarray().ravel()
    terms = vec.get_feature_names_out()
    top = row.argsort()[::-1][:n]
    top = [i for i in top if row[i] > 0]
    return pd.Series(row[top], index=terms[top], name="tfidf")


def transform_query(vec, text: str):
    """Vectorise un texte libre (utilisé par la recommandation / Streamlit)."""
    return vec.transform([clean_text(text)])


# --------------------------------------------------------------------------
# 6. Graphiques
# --------------------------------------------------------------------------
def _finish(fig, name: str, show: bool) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.show() if show else plt.close(fig)


def _label_mf(v) -> str:
    return "illimité" if pd.isna(v) else str(int(v))


def plot_experiments(exp: pd.DataFrame, show: bool = False) -> None:
    d = exp.copy()
    d["mf"] = d["max_features"].map(_label_mf)
    d["ng"] = d["ngram_min"].astype(str) + "-" + d["ngram_max"].astype(str)
    order = [_label_mf(v) for v in MAX_FEATURES_GRID]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, fmt, title, cmap in [
        (axes[0], "genre_coherence", ".3f", "Cohérence des genres des 5 voisins (plus haut = mieux)", "YlGn"),
        (axes[1], "vocab_size", "d", "Taille réelle du vocabulaire", "Blues"),
    ]:
        pv = d.pivot(index="ng", columns="mf", values=col).reindex(columns=order)
        sns.heatmap(pv, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar=False)
        ax.set(title=title, xlabel="max_features", ylabel="ngram_range")
    _finish(fig, "12_tfidf_experiments", show)


def plot_top_terms(top: pd.Series, show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top.values, y=top.index, color="#3b6ea5", ax=ax)
    ax.set(title="Termes au TF-IDF moyen le plus élevé", xlabel="TF-IDF moyen", ylabel="")
    _finish(fig, "13_tfidf_top_terms", show)


def plot_terms_by_genre(by_genre: dict, show: bool = False) -> None:
    names = list(by_genre)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, g in zip(axes.ravel(), names):
        ax.axis("off")
        ax.set_title(g, fontweight="bold")
        ax.text(0.5, 0.95, "\n".join(by_genre[g]), ha="center", va="top", fontsize=11, transform=ax.transAxes)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Termes les plus caractéristiques par genre", fontsize=13)
    _finish(fig, "14_tfidf_terms_by_genre", show)


# 8. Orchestration

def run_tfidf(skip_experiments: bool = False, df: pd.DataFrame | None = None, show: bool = False,
              include_keywords: bool = False) -> dict:
    df = load_features() if df is None else df
    genres = df["genres"].tolist()
    for folder in (REPORTS_DIR, MODELS_DIR, FEATURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # 1. Texte : valeurs manquantes -> "" 

    clean_overview = clean_series(build_text(df))
    clean = clean_series(build_text(df, include_keywords)) if include_keywords else clean_overview
    logger.info("Synopsis vides après nettoyage : %d", (clean_overview == "").sum())

    # 2. Configuration : meilleure combinaison max_features x ngram_range
    if skip_experiments:
        config = (DEFAULT_MAX_FEATURES, DEFAULT_NGRAM)
    else:
        exp = run_experiments(clean, genres)
        exp.to_csv(EXPERIMENTS_CSV, index=False)
        plot_experiments(exp, show)
        config = select_best(exp)

    # 3. Vectoriseur final et analyse de la matrice
    vec = make_vectorizer(*config)
    X = vec.fit_transform(clean).tocsr()
    top_genres = pd.Series([g for lst in genres for g in lst]).value_counts().head(6).index.tolist()
    info = {
        "config": config, "stats": matrix_stats(X),
        "coherence": genre_coherence(X, genres), "baseline": random_baseline(genres),
        "top_global": top_terms_global(vec, X, 20),
        "by_genre": top_terms_by_genre(vec, X, genres, top_genres),
        "vectorizer": vec, "X": X, "clean": clean,
    }
    plot_top_terms(info["top_global"], show)
    plot_terms_by_genre(info["by_genre"], show)

    # 4. Sauvegardes
    joblib.dump(vec, VECTORIZER_PATH, compress=3)
    sp.save_npz(MATRIX_PATH, X)
    df[["movie_id"]].to_csv(IDS_PATH, index=False)
    pd.DataFrame({
        "movie_id": df["movie_id"], "title": df["title"], "overview_clean": clean_overview,
        "n_tokens": clean_overview.str.split().str.len().fillna(0).astype(int),
        "has_text": (X.getnnz(axis=1) > 0).astype(int),
    }).to_csv(TEXT_CSV, index=False, encoding="utf-8")
    return info


def print_summary(info: dict) -> None:
    st = info["stats"]
    print(f"\nConfiguration retenue : max_features={info['config'][0]}, ngram_range={info['config'][1]}")
    print(f"Matrice : {st['n_docs']} x {st['vocab_size']} | creuse à {st['sparsity_pct']} % | {st['nnz_per_doc']} termes/film | {st['size_mb']} Mo")
    print(f"Cohérence des genres : {info['coherence']:.3f} (hasard {info['baseline']:.3f})")
    print("Top termes :", ", ".join(info["top_global"].index[:10]))
    for g, t in info["by_genre"].items():
        print(f"  {g:16s}: {', '.join(t)}")
    print(f"\nVectoriseur : {VECTORIZER_PATH}\nMatrice : {MATRIX_PATH}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-experiments", action="store_true")
    ap.add_argument("--keywords", action="store_true", help="vectoriser synopsis + mots-clés TMDB")
    args = ap.parse_args()
    print_summary(run_tfidf(skip_experiments=args.skip_experiments, include_keywords=args.keywords))

"""
Jour 4 - NLP : nettoyage du texte et TF-IDF sur `overview`.

    data/features/movies_features.csv -> data/features/movies_text.csv       (texte nettoyé)
                                      -> data/features/tfidf_matrix.npz      (matrice creuse)
                                      -> data/features/tfidf_movie_ids.csv   (ordre des lignes)
                                      -> models/tfidf_vectorizer.joblib      (vectoriseur ajusté)
                                      -> reports/tfidf_report.md, tfidf_experiments.csv
                                      -> reports/figures/12_, 13_, 14_*.png

Usage :
    python src/nlp/tfidf.py                       # expériences + vectoriseur final
    python src/nlp/tfidf.py --skip-experiments    # configuration par défaut (rapide)

DEUX USAGES DU TF-IDF, DEUX RÈGLES
----------------------------------
* Clustering / recommandation (non supervisé, pas de cible) : on ajuste le vectoriseur sur tous
  les films. C'est ce qui est fait ici.
* Classification (supervisé) : l'IDF est une statistique globale. Pour éviter le Data Leakage,
  le TfidfVectorizer sera mis DANS le Pipeline (Jour 5) et ajusté sur le train uniquement.
  Le nettoyage (clean_text) est sans état, ligne par ligne : il peut être fait avant le split.
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

# Mots trop génériques pour distinguer deux films + contractions (apostrophes retirées au nettoyage)
DOMAIN_STOPWORDS = {
    "film", "movie", "story", "dont", "doesnt", "didnt", "isnt", "wasnt", "cant", "couldnt",
    "wouldnt", "shouldnt", "arent", "werent", "hasnt", "havent", "hadnt", "im", "ive",
    "youre", "theyre", "thats", "theres",
}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS)

logger = logging.getLogger("tfidf")
sns.set_theme(style="whitegrid", context="notebook")


# --------------------------------------------------------------------------
# 1. Nettoyage du texte (sans état : ligne par ligne, aucune fuite possible)
# --------------------------------------------------------------------------
def clean_text(text) -> str:
    """minuscules, sans HTML/URL/chiffres/ponctuation, espaces normalisés. NaN -> ''."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"<[^>]+>", " ", t)                    # balises HTML
    t = re.sub(r"http\S+|www\.\S+", " ", t)           # URL
    t = re.sub(r"[’']s\b", "", t)                     # possessifs : "man's" -> "man"
    t = re.sub(r"[’']", "", t)                        # autres apostrophes : "don't" -> "dont"
    t = re.sub(r"[^a-zà-ÿ\s]", " ", t)                # tout sauf lettres (chiffres inclus)
    return re.sub(r"\s+", " ", t).strip()


def build_text(df: pd.DataFrame, include_keywords: bool = False) -> pd.Series:
    """Texte brut d'un film. Valeurs manquantes -> chaîne vide (jamais de NaN dans le vectoriseur)."""
    text = df["overview"].fillna("").astype(str)
    if include_keywords:
        text = text + " " + df["keywords"].apply(lambda k: " ".join(k))
    return text


def clean_series(texts: pd.Series) -> pd.Series:
    return texts.map(clean_text)


# --------------------------------------------------------------------------
# 2. Vectoriseur
# --------------------------------------------------------------------------
def make_vectorizer(max_features=DEFAULT_MAX_FEATURES, ngram_range=DEFAULT_NGRAM,
                    min_df=MIN_DF, max_df=MAX_DF) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words=STOP_WORDS,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,      # ignore les termes présents dans moins de 2 films (fautes, noms rares)
        max_df=max_df,      # ignore les termes présents dans plus de 80 % des films
        lowercase=False,    # déjà fait par clean_text
        norm="l2",          # vecteurs de norme 1 : produit scalaire = similarité cosinus
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


# --------------------------------------------------------------------------
# 3. Évaluation sans étiquette : cohérence des genres des voisins
# --------------------------------------------------------------------------
def _jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def genre_coherence(X, genres: list, k: int = 5) -> float:
    """
    Pour chaque film, prend ses k voisins les plus proches (cosinus) et mesure le
    chevauchement de genres (Jaccard). Plus c'est haut, plus le TF-IDF capte du sens.
    Les genres ne servent qu'à ÉVALUER, jamais comme entrée : pas de leakage.
    """
    sets = [set(g) for g in genres]
    sim = linear_kernel(X)                      # X est normalisé L2 : produit scalaire = cosinus
    np.fill_diagonal(sim, -1.0)                 # un film n'est pas son propre voisin
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
    """Chevauchement moyen de genres entre deux films tirés au hasard (point de comparaison)."""
    rng = np.random.default_rng(seed)
    sets = [set(g) for g in genres]
    i = rng.integers(0, len(sets), n_pairs)
    j = rng.integers(0, len(sets), n_pairs)
    return float(np.mean([_jaccard(sets[a], sets[b]) for a, b in zip(i, j) if a != b]))


# --------------------------------------------------------------------------
# 4. Expériences max_features x ngram_range
# --------------------------------------------------------------------------
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
    df["max_features"] = df["max_features"].astype("Int64")  # None -> <NA>
    return df


def select_best(exp: pd.DataFrame, tolerance: float = 0.005):
    """Meilleure cohérence ; à égalité (± tolérance) on préfère le plus petit vocabulaire."""
    ok = exp[exp["genre_coherence"] >= exp["genre_coherence"].max() - tolerance]
    row = ok.sort_values(["vocab_size", "ngram_max"]).iloc[0]
    mf = None if pd.isna(row["max_features"]) else int(row["max_features"])
    return mf, (int(row["ngram_min"]), int(row["ngram_max"]))


# --------------------------------------------------------------------------
# 5. Termes représentatifs
# --------------------------------------------------------------------------
def top_terms_global(vec, X, n: int = 20) -> pd.Series:
    means = np.asarray(X.mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    idx = means.argsort()[::-1][:n]
    return pd.Series(means[idx], index=terms[idx], name="tfidf_moyen")


def top_terms_by_genre(vec, X, genres: list, genre_names: list, n: int = 8, min_docs: int = 5) -> dict:
    """Termes les plus CARACTÉRISTIQUES d'un genre : moyenne dans le genre moins moyenne globale."""
    terms = vec.get_feature_names_out()
    glob = np.asarray(X.mean(axis=0)).ravel()
    out = {}
    for g in genre_names:
        mask = np.array([g in lst for lst in genres])
        if mask.sum() < min_docs:  # genre trop rare pour être représentatif
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


# --------------------------------------------------------------------------
# 7. Rapport
# --------------------------------------------------------------------------
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def _effect(delta: float) -> str:
    if delta > 0.003:
        return "améliore"
    if delta < -0.003:
        return "dégrade"
    return "ne change pratiquement pas"


def write_report(info: dict) -> None:
    exp = info.get("experiments")
    L = ["# Rapport TF-IDF (Jour 4)", "",
         "## 1. Qualité du texte", "",
         f"- Films : {info['n_docs']}",
         f"- Synopsis vides ou sans mot exploitable après nettoyage : {info['n_empty']} ({info['n_empty'] / info['n_docs'] * 100:.1f} %)",
         f"- Longueur médiane avant nettoyage : {info['words_raw_median']:.0f} mots ; après nettoyage et retrait des mots vides : {info['words_clean_median']:.0f} mots",
         f"- Synopsis très courts (moins de 10 mots nettoyés) : {info['n_short']}", ""]
    if exp is not None:
        base = info["baseline"]
        L += ["## 2. Expériences max_features x ngram_range", "",
              f"min_df={MIN_DF}, max_df={MAX_DF}. La cohérence mesure le chevauchement de genres (Jaccard) entre un film et ses "
              f"5 voisins ; le hasard donne {base:.3f}.", "",
              md_table(exp), ""]
        by_ng = exp.groupby("ngram_max")["genre_coherence"].mean()
        mf_num = exp.dropna(subset=["max_features"])
        by_mf = mf_num.groupby("max_features")["genre_coherence"].mean()
        L += ["**Lecture :**",
              f"- Unigrammes -> bigrammes : {_effect(by_ng.get(2, np.nan) - by_ng.get(1, np.nan))} la cohérence "
              f"({by_ng.get(1, float('nan')):.3f} -> {by_ng.get(2, float('nan')):.3f} en moyenne sur les max_features).",
              f"- Bigrammes -> trigrammes : {_effect(by_ng.get(3, np.nan) - by_ng.get(2, np.nan))} la cohérence "
              f"({by_ng.get(2, float('nan')):.3f} -> {by_ng.get(3, float('nan')):.3f}).",
              f"- max_features de {int(by_mf.index.min())} à {int(by_mf.index.max())} : {_effect(by_mf.iloc[-1] - by_mf.iloc[0])} "
              f"la cohérence ({by_mf.iloc[0]:.3f} -> {by_mf.iloc[-1]:.3f}), au prix d'une matrice plus grande et d'un calcul plus long.", ""]
    st = info["stats"]
    mf, ng = info["config"]
    L += ["## 3. Configuration retenue et matrice finale", "",
          f"- `max_features={mf}`, `ngram_range={ng}`, `min_df={MIN_DF}`, `max_df={MAX_DF}`, `norm='l2'`",
          f"- Dimensions : **{st['n_docs']} films x {st['vocab_size']} termes**",
          f"- Valeurs non nulles : {st['nnz']} -> matrice creuse à **{st['sparsity_pct']} %** de zéros",
          f"- En moyenne {st['nnz_per_doc']} termes actifs par film sur {st['vocab_size']}",
          f"- Mémoire (format creux) : {st['size_mb']} Mo ; en format dense elle ferait environ "
          f"{st['n_docs'] * st['vocab_size'] * 8 / 1e6:.0f} Mo",
          f"- Vocabulaire : {info['n_uni']} unigrammes et {info['n_bi']} n-grammes de plus d'un mot",
          f"- Cohérence des genres : {info['coherence']:.3f} (hasard : {info['baseline']:.3f}, soit x{info['coherence'] / max(info['baseline'], 1e-9):.1f})", "",
          "## 4. Termes représentatifs", "",
          "### Les plus élevés en moyenne", "", ", ".join(f"`{t}`" for t in info["top_global"].index[:20]), "",
          "### Les plus caractéristiques par genre (écart à la moyenne globale)", ""]
    L += [f"- **{g}** : {', '.join(t)}" for g, t in info["by_genre"].items()]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# 8. Orchestration
# --------------------------------------------------------------------------
def run_tfidf(skip_experiments: bool = False, df: pd.DataFrame | None = None, show: bool = False) -> dict:
    df = load_features() if df is None else df
    raw = build_text(df)
    clean = clean_series(raw)
    genres = df["genres"].tolist()

    # -- Qualité du texte
    tokens_clean = clean.str.split().str.len().fillna(0)
    info = {
        "n_docs": len(df),
        "n_empty": int((clean == "").sum()),
        "words_raw_median": float(raw.str.split().str.len().median()),
        "words_clean_median": float(tokens_clean.median()),
        "n_short": int((tokens_clean < 10).sum()),
        "baseline": random_baseline(genres),
    }

    # -- Expériences
    if skip_experiments:
        config = (DEFAULT_MAX_FEATURES, DEFAULT_NGRAM)
        info["experiments"] = None
    else:
        exp = run_experiments(clean, genres)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        exp.to_csv(EXPERIMENTS_CSV, index=False)
        plot_experiments(exp, show)
        info["experiments"] = exp
        config = select_best(exp)
    info["config"] = config

    # -- Vectoriseur final
    vec = make_vectorizer(*config)
    X = vec.fit_transform(clean).tocsr()
    terms = vec.get_feature_names_out()
    info["stats"] = matrix_stats(X)
    info["n_bi"] = int(sum(" " in t for t in terms))
    info["n_uni"] = len(terms) - info["n_bi"]
    info["coherence"] = genre_coherence(X, genres)
    info["top_global"] = top_terms_global(vec, X, 20)
    top_genres = pd.Series([g for lst in genres for g in lst]).value_counts().head(6).index.tolist()
    info["by_genre"] = top_terms_by_genre(vec, X, genres, top_genres)
    plot_top_terms(info["top_global"], show)
    plot_terms_by_genre(info["by_genre"], show)

    # -- Sauvegardes
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, VECTORIZER_PATH, compress=3)
    sp.save_npz(MATRIX_PATH, X)
    df[["movie_id"]].to_csv(IDS_PATH, index=False)
    pd.DataFrame({
        "movie_id": df["movie_id"], "title": df["title"],
        "overview_clean": clean, "n_tokens": tokens_clean.astype(int),
        "has_text": (X.getnnz(axis=1) > 0).astype(int),
    }).to_csv(TEXT_CSV, index=False, encoding="utf-8")
    write_report(info)
    info["vectorizer"], info["X"], info["clean"] = vec, X, clean
    return info


def print_summary(info: dict) -> None:
    st = info["stats"]
    print(f"\nConfiguration retenue : max_features={info['config'][0]}, ngram_range={info['config'][1]}")
    print(f"Matrice : {st['n_docs']} x {st['vocab_size']} | creuse à {st['sparsity_pct']} % | {st['nnz_per_doc']} termes/film | {st['size_mb']} Mo")
    print(f"Cohérence des genres : {info['coherence']:.3f} (hasard {info['baseline']:.3f})")
    print("Top termes :", ", ".join(info["top_global"].index[:10]))
    for g, t in info["by_genre"].items():
        print(f"  {g:16s}: {', '.join(t)}")
    print(f"\nRapport : {REPORT_MD}\nVectoriseur : {VECTORIZER_PATH}\nMatrice : {MATRIX_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-experiments", action="store_true")
    args = ap.parse_args()
    print_summary(run_tfidf(skip_experiments=args.skip_experiments))

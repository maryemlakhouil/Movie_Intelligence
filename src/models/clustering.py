import argparse
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features.build_features import FEATURES_DIR, load_features 
from src.models.classification import THRESHOLD 
from src.nlp.tfidf import IDS_PATH, MATRIX_PATH, VECTORIZER_PATH 

REPORTS_DIR, MODELS_DIR, FIG_DIR = PROJECT_ROOT / "reports", PROJECT_ROOT / "models", PROJECT_ROOT / "reports" / "figures"
RANDOM_STATE, K_RANGE, K_MIN_PROFILE = 42, range(2, 11), 3
STRUCT_COLS = ["release_year", "runtime", "budget_log", "vote_average", "vote_count_log", "n_genres"]


# Outils K-Means

def silhouette_by_k(X, ks=K_RANGE) -> dict:
    return {k: silhouette_score(X, KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)) for k in ks}

# K au meilleur silhouette
def pick_k(scores: dict, k_min: int = K_MIN_PROFILE) -> int:
    return max((k for k in scores if k >= k_min), key=scores.get)


def top_genres(lists, n=3) -> str:
    return ", ".join(g for g, _ in Counter(x for lst in lists for x in lst).most_common(n))


# Interprétation des clusters

def structured_profile(df: pd.DataFrame, labels) -> pd.DataFrame:
    d = df.assign(cluster=labels, fort_engagement=df["vote_count"] >= THRESHOLD)
    g = d.groupby("cluster")
    prof = g.agg(films=("movie_id", "size"), annee=("release_year", "median"), duree=("runtime", "median"),
                 budget=("budget", "median"), note=("vote_average", "mean"), votes=("vote_count", "median"),
                 nb_genres=("n_genres", "mean"), pct_fort_engagement=("fort_engagement", "mean"))
    prof["pct_fort_engagement"] *= 100
    prof["top_genres"] = g["genres"].apply(top_genres)
    return prof.round(1)


def text_profile(X, labels, vec, df: pd.DataFrame, n_terms=8) -> pd.DataFrame:

    terms, glob, rows = vec.get_feature_names_out(), np.asarray(X.mean(axis=0)).ravel(), []
    for c in sorted(set(labels)):
        m = labels == c
        diff = np.asarray(X[m].mean(axis=0)).ravel() - glob
        rows.append({"cluster": c, "films": int(m.sum()), "termes": ", ".join(terms[diff.argsort()[::-1][:n_terms]]),
                     "top_genres": top_genres(df.loc[m, "genres"])})
    return pd.DataFrame(rows).set_index("cluster")


# --------------------------------------------------------------------------
# Graphiques
# --------------------------------------------------------------------------
def plot_silhouettes(sc_a: dict, sc_b: dict, k_a: int, k_b: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, sc, k, title in [(axes[0], sc_a, k_a, "Variables structurées"), (axes[1], sc_b, k_b, "TF-IDF (SVD)")]:
        ax.plot(list(sc), list(sc.values()), "o-", color="#3b6ea5")
        ax.axvline(k, color="red", ls="--", label=f"K retenu = {k}")
        ax.set(title=f"Silhouette Score : {title}", xlabel="K", ylabel="silhouette")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "19_silhouette_scores.png", dpi=150)
    plt.close(fig)


def plot_clusters_2d(Xa, la, Xb, lb) -> None:
    pa, pb = PCA(2, random_state=RANDOM_STATE), TruncatedSVD(2, random_state=RANDOM_STATE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, proj, X, lab, title in [(axes[0], pa, Xa, la, "Structurées (PCA 2D)"),
                                    (axes[1], pb, Xb, lb, "TF-IDF (TruncatedSVD 2D)")]:
        P = proj.fit_transform(X)
        sc = ax.scatter(P[:, 0], P[:, 1], c=lab, cmap="tab10", s=12, alpha=0.7)
        ax.set(title=f"Clusters : {title}", xlabel=f"axe 1 ({proj.explained_variance_ratio_[0]:.0%})",
               ylabel=f"axe 2 ({proj.explained_variance_ratio_[1]:.0%})")
        ax.legend(*sc.legend_elements(), title="cluster", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "20_clusters_2d.png", dpi=150)
    plt.close(fig)

# Programme principal

def main(k_struct=None, k_text=None) -> None:
    df = load_features()
    df["vote_count_log"] = np.log1p(df["vote_count"])
    X_tfidf = sp.load_npz(MATRIX_PATH).tocsr()
    if not (pd.read_csv(IDS_PATH)["movie_id"].values == df["movie_id"].values).all():
        raise SystemExit("La matrice TF-IDF n'est pas alignée sur les films : relancer src/nlp/tfidf.py")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # A) variables structurées : imputation médiane + standardisation (K-Means utilise des distances)
    prep_a = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    Xa = prep_a.fit_transform(df[STRUCT_COLS])
    sc_a = silhouette_by_k(Xa)
    k_a = k_struct or pick_k(sc_a)
    km_a = Pipeline([("prep", prep_a), ("km", KMeans(k_a, n_init=10, random_state=RANDOM_STATE))]).fit(df[STRUCT_COLS])
    lab_a = km_a.predict(df[STRUCT_COLS])

    # B) texte : TF-IDF -> SVD (réduction) -> normalisation -> K-Means
    n_comp = min(50, X_tfidf.shape[1] - 1)
    prep_b = Pipeline([("svd", TruncatedSVD(n_comp, random_state=RANDOM_STATE)), ("norm", Normalizer(copy=False))])
    Xb = prep_b.fit_transform(X_tfidf)
    sc_b = silhouette_by_k(Xb)
    k_b = k_text or pick_k(sc_b)
    km_b = Pipeline([("prep", prep_b), ("km", KMeans(k_b, n_init=10, random_state=RANDOM_STATE))]).fit(X_tfidf)
    lab_b = km_b.predict(X_tfidf)

    print("=== Silhouette par K ===")
    print(pd.DataFrame({"structurées": sc_a, "TF-IDF": sc_b}).round(3).to_string())
    print(f"\nK retenu : structurées = {k_a}, TF-IDF = {k_b}")

    prof_a = structured_profile(df, lab_a)
    prof_b = text_profile(X_tfidf, lab_b, joblib.load(VECTORIZER_PATH), df)
    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print("\n=== Profils des clusters (variables structurées) ===\n", prof_a.to_string())
    print("\n=== Profils des clusters (texte) ===\n", prof_b.to_string())

    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    prof_a.to_csv(REPORTS_DIR / "cluster_profiles_structured.csv")
    prof_b.to_csv(REPORTS_DIR / "cluster_profiles_text.csv")
    pd.DataFrame({"movie_id": df["movie_id"], "cluster_structured": lab_a, "cluster_text": lab_b}).to_csv(
        FEATURES_DIR / "movie_clusters.csv", index=False)
    joblib.dump(km_a, MODELS_DIR / "kmeans_structured.joblib", compress=3)
    joblib.dump(km_b, MODELS_DIR / "kmeans_text.joblib", compress=3)
    plot_silhouettes(sc_a, sc_b, k_a, k_b)
    plot_clusters_2d(Xa, lab_a, Xb, lab_b)
    print(f"\nSilhouette final : structurées = {sc_a[k_a]:.3f}, TF-IDF = {sc_b[k_b]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-structured", type=int)
    ap.add_argument("--k-text", type=int)
    a = ap.parse_args()
    main(a.k_structured, a.k_text)

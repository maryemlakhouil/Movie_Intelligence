import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features.build_features import load_features  
from src.nlp.tfidf import IDS_PATH, MATRIX_PATH, VECTORIZER_PATH, genre_coherence, random_baseline, transform_query  

# df : une ligne par film ; X : matrice TF-IDF 

class Recommender:

    def __init__(self, df: pd.DataFrame, X):
        self.df = df.reset_index(drop=True)
        self.X = X.tocsr()

    @classmethod
    def load(cls):
        df = load_features()
        X = sp.load_npz(MATRIX_PATH)
        if not (pd.read_csv(IDS_PATH)["movie_id"].values == df["movie_id"].values).all():
            raise SystemExit("Matrice TF-IDF non alignée sur les films : relancer src/nlp/tfidf.py")
        return cls(df, X)
    
    # Indice du film : titre exact, sinon titre contenant le texte.

    def find_movie(self, query: str) -> int:
        titles, q = self.df["title"].astype(str).str.lower(), query.strip().lower()
        hits = self.df.index[titles == q]
        if len(hits) == 0:
            hits = self.df.index[titles.str.contains(q, regex=False)]
        if len(hits) == 0:
            raise ValueError(f"Film introuvable : {query!r}")
        return int(self.df.loc[hits, "vote_count"].idxmax())

    def _top(self, scores: np.ndarray, n: int, exclude: int | None = None) -> pd.DataFrame:
        scores = scores.copy()
        if exclude is not None:
            scores[exclude] = -1                      
        idx = np.argsort(-scores)[:n]
        d = self.df.loc[idx]
        return pd.DataFrame({
            "title": d["title"].values,
            "genres": [", ".join(g) for g in d["genres"]],
            "year": d["release_year"].astype("Int64").values,
            "similarity": scores[idx].round(3),
        })
    # Les n films les plus proches (cosinus) du film situé à la ligne i

    def recommend_by_index(self, i: int, n: int = 5) -> pd.DataFrame:
        if self.X[i].nnz == 0:
            raise ValueError("Ce film n'a pas de synopsis exploitable.")
        return self._top(cosine_similarity(self.X[i], self.X).ravel(), n, exclude=i)

    def recommend(self, title: str, n: int = 5) -> pd.DataFrame:
        return self.recommend_by_index(self.find_movie(title), n)
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--title")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()

    rec = Recommender.load()
    i = rec.find_movie(a.title) if a.title else int(rec.df["vote_count"].idxmax())
    row = rec.df.loc[i]
    print(f"Film choisi : {row['title']} ({int(row['release_year'])}) | genres : {', '.join(row['genres'])}\n")
    print(rec.recommend(row["title"], a.n).to_string(index=False))

    if a.evaluate:
        c, b = genre_coherence(rec.X, rec.df["genres"].tolist(), k=a.n), random_baseline(rec.df["genres"].tolist())
        print(f"\nCohérence des genres des {a.n} recommandations : {c:.3f} (hasard : {b:.3f}, soit x{c / b:.1f})")

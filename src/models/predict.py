import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.cleaning.clean_data import MIN_MONEY  
from src.features.build_features import build_features  
from src.nlp.tfidf import clean_text 


def make_input_row(runtime, year, month, budget, language, genres, overview, n_keywords,
                   title: str = "Nouveau film") -> pd.DataFrame:
    raw = pd.DataFrame([{
        "movie_id": 0, "title": title, "overview": overview or None,
        "release_date": pd.Timestamp(year=int(year), month=int(month), day=15),
        "runtime": float(runtime), "original_language": language, "status": "Released",
        "genres": list(genres), "keywords": ["keyword"] * int(n_keywords),
        "budget": float(budget) if budget and budget >= MIN_MONEY else np.nan,   
        "revenue": np.nan, "popularity": np.nan, "vote_average": np.nan, "vote_count": np.nan,
    }])
    row = build_features(raw)
    row["overview_clean"] = clean_text(overview)
    return row


def predict_engagement(pipe, row: pd.DataFrame) -> tuple[int, float | None]:
    cols = list(pipe.named_steps["prep"].feature_names_in_)      
    X = row.reindex(columns=cols)                                 
    genre_cols = [c for c in cols if c.startswith("genre_")]
    X[genre_cols] = X[genre_cols].fillna(0)                      
    label = int(pipe.predict(X)[0])
    proba = float(pipe.predict_proba(X)[0, 1]) if hasattr(pipe, "predict_proba") else None
    return label, proba

# chargement avec cache dess données et modèles

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS / "figures"
FEATURES_CSV = FEATURES_DIR / "movies_features.csv"
CLUSTERS_CSV = FEATURES_DIR / "movie_clusters.csv"
CLASSIFIER_PATH = MODELS_DIR / "best_classifier.joblib"
KMEANS_TEXT_PATH = MODELS_DIR / "kmeans_text.joblib"

THRESHOLD = 4000  

CLUSTER_NAMES = {
    "structured": {0: "Classiques bien notés", 1: "Sorties récentes à faible audience", 2: "Blockbusters récents multi-genres"},
    "text": {},  
}

# Arrête la page si le pipeline n'a pas encore produit les fichiers

def require(*paths) -> None:
    missing = [Path(p).name for p in paths if not Path(p).exists()]
    if missing:
        st.error(f"Fichiers manquants : {', '.join(missing)}. Lancez d'abord le DAG `movie_pipeline` dans Airflow "
                 "(ou `python -m src.pipeline`).")
        st.stop()


def histogram(series: pd.Series, bins: int = 20) -> pd.Series:
    counts, edges = np.histogram(series.dropna(), bins=bins)
    return pd.Series(counts, index=np.round(edges[:-1], 2), name="films")


@st.cache_data
def load_movies() -> pd.DataFrame:
    from src.features.build_features import load_features
    return load_features()


@st.cache_data
def load_clusters() -> pd.DataFrame:
    return pd.read_csv(CLUSTERS_CSV)


@st.cache_resource
def load_classifier():
    return joblib.load(CLASSIFIER_PATH)


@st.cache_resource
def load_kmeans_text():
    return joblib.load(KMEANS_TEXT_PATH)


@st.cache_resource
def load_vectorizer():
    from src.nlp.tfidf import VECTORIZER_PATH
    return joblib.load(VECTORIZER_PATH)


@st.cache_resource
def load_recommender():
    from src.recommendation.recommender import Recommender
    return Recommender.load()

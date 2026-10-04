"""
Jour 2 - Chargement des films nettoyés dans MongoDB.

    data/processed/movies_clean.csv  ->  MongoDB  (base movie_db, collection movies)

Usage :
    python src/database/mongo_loader.py

Le chargement est IDEMPOTENT : on utilise un upsert (ReplaceOne, upsert=True) avec
_id = movie_id. Relancer le script met à jour les films existants sans créer de doublons,
ce qui est indispensable pour l'automatisation Airflow (Jour 9).
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient, ReplaceOne
from pymongo.errors import PyMongoError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.cleaning.clean_data import CLEAN_CSV, LIST_COLS, load_clean  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "movie_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "movies")

INT_COLS = {"movie_id", "runtime", "budget", "revenue", "vote_count"}
logger = logging.getLogger("mongo")


def get_collection(uri: str = MONGO_URI):
    """Connexion + test de disponibilité (échec rapide si MongoDB n'est pas lancé)."""
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[MONGO_DB][MONGO_COLLECTION]


def _to_native(key: str, value):
    """Convertit les types pandas/numpy en types Python compris par MongoDB."""
    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, (list, str)) and pd.isna(value)):
        return None  # NaN / NaT -> null
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return int(value) if key in INT_COLS and value.is_integer() else value
    return value


def df_to_documents(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> liste de documents MongoDB (_id = movie_id)."""
    docs = []
    for rec in df.to_dict(orient="records"):
        doc = {k: _to_native(k, v) for k, v in rec.items()}
        doc["_id"] = doc["movie_id"]
        docs.append(doc)
    return docs


def create_indexes(col) -> None:
    col.create_index([("genres", ASCENDING)])
    col.create_index([("original_language", ASCENDING)])
    col.create_index([("release_date", ASCENDING)])
    col.create_index([("vote_average", DESCENDING)])
    col.create_index([("popularity", DESCENDING)])


def load_to_mongo(csv_path: Path = CLEAN_CSV) -> int:
    df = load_clean(csv_path)
    docs = df_to_documents(df)
    col = get_collection()
    ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs]
    result = col.bulk_write(ops, ordered=False)
    create_indexes(col)
    logger.info(
        "Upsert terminé : %d insérés, %d modifiés | total en base : %d",
        result.upserted_count, result.modified_count, col.count_documents({}),
    )
    return col.count_documents({})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        load_to_mongo()
    except PyMongoError as exc:
        raise SystemExit(f"MongoDB inaccessible ({MONGO_URI}) : {exc}\nLancer : docker compose up -d mongo")
    except FileNotFoundError:
        raise SystemExit("movies_clean.csv introuvable : lancer d'abord src/cleaning/clean_data.py")

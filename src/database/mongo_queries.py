import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.database.mongo_loader import get_collection  

SHOW = {"_id": 0, "title": 1, "release_date": 1, "vote_average": 1, "vote_count": 1, "popularity": 1}

# les champs a afficher 

def show(title: str, docs) -> None:
    print(f"\n### {title}")
    docs = list(docs)
    if not docs:
        print("  (aucun résultat)")
    for d in docs:
        if "release_date" in d and d["release_date"]:
            d["release_date"] = d["release_date"].strftime("%Y-%m-%d")
        print("  ", d)


# ------------------------------ Requêtes simples ------------------------------

# les films les plus populaires

def q1_top_popular(col, n=5):
    return col.find({}, SHOW).sort("popularity", -1).limit(n)

# bien notés ET beaucoup votés 
def q2_acclaimed(col):
    return col.find(
        {"vote_average": {"$gte": 8.0}, "vote_count": {"$gte": 1000}}, SHOW).sort("vote_average", -1).limit(5)


def q3_short_comedies(col):
    return col.find({"genres": "Comedy", "runtime": {"$lt": 90}}, SHOW).sort("popularity", -1).limit(5)

# films sortis depuis 2023 

def q4_recent(col):
    return col.find({"release_date": {"$gte": datetime(2023, 1, 1)}}, SHOW).sort("vote_count", -1).limit(5)

# all : films qui sont à la fois Action ET Science Fiction

def q5_multi_genre(col):
    return col.find({"genres": {"$all": ["Action", "Science Fiction"]}}, SHOW).sort("vote_average", -1).limit(5)

# maj et min 
def q6_title_search(col):
    return col.find({"title": {"$regex": "star", "$options": "i"}}, SHOW).limit(5)


def q7_stats(col):
    return [
        {"films_total": col.count_documents({})},
        {"films_avec_budget": col.count_documents({"budget": {"$ne": None}})},
        {"langues_distinctes": len(col.distinct("original_language"))},
    ]


# ------------------------------ Agrégations ------------------------------
def agg_genres(col):
    pipeline = [
        {"$match": {"vote_count": {"$gte": 50}}},
        {"$unwind": "$genres"}, # join unnset
        {"$group": {
            "_id": "$genres",
            "n_films": {"$sum": 1},
            "avg_vote": {"$avg": "$vote_average"},
            "avg_popularity": {"$avg": "$popularity"},
            "avg_runtime": {"$avg": "$runtime"},
        }},
        {"$sort": {"n_films": -1}},
        {"$project": {
            "_id": 0, "genre": "$_id", "n_films": 1,
            "avg_vote": {"$round": ["$avg_vote", 2]},
            "avg_popularity": {"$round": ["$avg_popularity", 1]},
            "avg_runtime": {"$round": ["$avg_runtime", 0]},
        }},
    ]
    return col.aggregate(pipeline)


def agg_decades(col):
    pipeline = [
        {"$match": {"release_date": {"$ne": None}}},
        {"$project": {
            "decade": {"$multiply": [{"$floor": {"$divide": [{"$year": "$release_date"}, 10]}}, 10]},
            "vote_average": 1, "popularity": 1, "vote_count": 1,
        }},
        {"$group": {
            "_id": "$decade",
            "n_films": {"$sum": 1},
            "avg_vote": {"$avg": "$vote_average"},
            "avg_popularity": {"$avg": "$popularity"},
        }},
        {"$sort": {"_id": 1}},
        {"$project": {
            "_id": 0, "decade": "$_id", "n_films": 1,
            "avg_vote": {"$round": ["$avg_vote", 2]},
            "avg_popularity": {"$round": ["$avg_popularity", 1]},
        }},
    ]
    return col.aggregate(pipeline)

# mots-clés les plus fréquents

def agg_keywords(col, n=15):
    pipeline = [
        {"$unwind": "$keywords"},
        {"$group": {"_id": "$keywords", "n_films": {"$sum": 1}}},
        {"$sort": {"n_films": -1}},
        {"$limit": n},
        {"$project": {"_id": 0, "keyword": "$_id", "n_films": 1}},
    ]
    return col.aggregate(pipeline)

# group par langue originale avec note moyenne

def agg_languages(col):
    pipeline = [
        {"$group": {"_id": "$original_language", "n_films": {"$sum": 1}, "avg_vote": {"$avg": "$vote_average"}}},
        {"$sort": {"n_films": -1}},
        {"$limit": 8},
        {"$project": {"_id": 0, "language": "$_id", "n_films": 1, "avg_vote": {"$round": ["$avg_vote", 2]}}},
    ]
    return col.aggregate(pipeline)


if __name__ == "__main__":
    col = get_collection()
    show("Q1 - Top 5 popularité", q1_top_popular(col))
    show("Q2 - Films acclamés (note >= 8, votes >= 1000)", q2_acclaimed(col))
    show("Q3 - Comédies de moins de 90 min", q3_short_comedies(col))
    show("Q4 - Sorties depuis 2023", q4_recent(col))
    show("Q5 - Action ET Science Fiction", q5_multi_genre(col))
    show("Q6 - Titres contenant 'star'", q6_title_search(col))
    show("Q7 - Statistiques de la collection", q7_stats(col))
    show("AGG 1 - Bilan par genre", agg_genres(col))
    show("AGG 2 - Bilan par décennie", agg_decades(col))
    show("AGG 3 - Top 15 mots-clés", agg_keywords(col))
    show("AGG 4 - Films par langue originale", agg_languages(col))

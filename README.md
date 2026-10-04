# Movie Recommender (TMDB)

Projet : extraction TMDB, EDA, classification, clustering, recommandation, Streamlit, Airflow, Docker.

## Démarrage (Jour 1)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
# 1. Renseigner TMDB_READ_TOKEN dans .env
python src/extraction/extract_tmdb.py --n-movies 40     # test
python src/extraction/extract_tmdb.py --n-movies 2000   # extraction complète
python src/extraction/check_raw.py                      # contrôle qualité
```

## Jour 2 : nettoyage + MongoDB
```bash
pip install -r requirements.txt
python src/cleaning/clean_data.py            # -> data/processed/movies_clean.csv + reports/cleaning_report.txt
docker compose up -d mongo                   # (ou: podman run -d -p 27017:27017 --name movie_mongo docker.io/library/mongo:7)
python src/database/mongo_loader.py          # chargement (upsert, relançable)
python src/database/mongo_queries.py         # requêtes + agrégations
python -m pytest                             # tests du nettoyage
```

## Avancement
- [x] J1 API + extraction + raw JSON
- [x] J2 Nettoyage + MongoDB
- [ ] J3 EDA + Feature Engineering
- [ ] J4 NLP TF-IDF
- [ ] J5 Classification
- [ ] J6 Cross-validation + GridSearch
- [ ] J7 K-Means
- [ ] J8 Recommandation
- [ ] J9 Airflow + Docker
- [ ] J10 Streamlit + tests

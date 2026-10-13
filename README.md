# Movie Intelligence : analyse, classification, clustering et recommandation de films (TMDB)

Pipeline complet de data science sur ~2 000 films de l'API TMDB : extraction, nettoyage, MongoDB, EDA, feature engineering,
NLP (TF-IDF), classification de l'engagement, clustering, recommandation, automatisation Airflow, conteneurisation Docker
et application Streamlit.

## Démarrage rapide (Docker)

```bash
cp .env.example .env                              # puis renseigner TMDB_READ_TOKEN dans .env
echo "AIRFLOW_UID=$(id -u)" >> .env               # une seule fois
docker compose up -d --build                      # MongoDB + Airflow + Streamlit
```
1. **Airflow** : http://localhost:8080 -> DAG `movie_pipeline` -> activer, puis lancer (▶) :
   ```bash
   docker compose exec airflow airflow dags unpause movie_pipeline
   docker compose exec airflow airflow dags trigger movie_pipeline
   ```
2. **Application** : http://localhost:8501 (disponible une fois le pipeline terminé).
3. Arrêt : `docker compose down` (les données restent dans `data/`, `models/`, `reports/`).

Sans Docker : `pip install -r requirements.txt`, puis `python -m src.pipeline` (ou `python -m src.pipeline clean features ...`)
et `streamlit run app/streamlit_app.py`. Les modèles `.joblib` doivent être lus avec la même version de scikit-learn que celle
qui les a créés : le plus simple est de tout faire dans Docker, ou tout hors Docker.

## Pipeline (DAG `movie_pipeline`)

`extract -> clean -> features -> mongo -> tfidf -> classifier -> clustering`

| Étape | Code | Sorties principales |
|---|---|---|
| Extraction TMDB | `src/extraction/extract_tmdb.py` | `data/raw/` (JSON bruts, cache local) |
| Nettoyage Pandas | `src/cleaning/clean_data.py` | `data/processed/movies_clean.csv`, `reports/cleaning_report.txt` |
| Feature engineering | `src/features/build_features.py` | `data/features/movies_features.csv` |
| MongoDB (upsert, requêtes, agrégations) | `src/database/` | collection `movie_db.movies` |
| TF-IDF | `src/nlp/tfidf.py` | `models/tfidf_vectorizer.joblib`, `data/features/tfidf_matrix.npz` |
| Classification (CV + GridSearch) | `src/models/classification.py`, `validation.py` | `models/best_classifier.joblib`, `reports/cv_results.csv` |
| Clustering K-Means | `src/models/clustering.py` | `models/kmeans_*.joblib`, `data/features/movie_clusters.csv` |
| Recommandation | `src/recommendation/recommender.py` | (utilisée par l'application) |

Hors DAG : l'EDA (`python src/eda/eda_plots.py` -> `reports/figures/`) et la classification détaillée
(`python src/models/classification.py` : 3 modèles, baseline, démonstration de Data Leakage).

## Choix méthodologiques importants

- **Cible** : `high_engagement` = `vote_count >= 4000` (seuil fixe proche de la médiane, défini avant le split).
- **Data Leakage évité** : `vote_count`, `popularity`, `vote_average`, `revenue` ne sont jamais des features ; imputation, mise à
  l'échelle, encodage et TF-IDF sont dans un `Pipeline` ajusté sur le train ; aucune imputation avant le split.
- **Validation** : `StratifiedKFold` (5 plis) + `cross_validate`, puis `GridSearchCV` ; le jeu de test n'est utilisé qu'à la fin.
- **Recommandation** : TF-IDF sur synopsis + mots-clés TMDB (noms de genres retirés pour une évaluation non circulaire),
  similarité cosinus, évaluation par cohérence des genres comparée au hasard.

## Application Streamlit

| Page | Contenu |
|---|---|
| Accueil | Indicateurs, filtres (années, genres), distributions, genres, budget/revenus, figures de l'EDA |
| Classification | Prédiction de l'engagement d'un film saisi, performances des modèles |
| Clusters | Profils (variables structurées et texte), exploration, cluster d'un synopsis |
| Recommandation | 5 films similaires (titre, genres, année, similarité) à partir d'un film ou d'une description |

Les noms des clusters se renseignent dans `app/shared.py` (`CLUSTER_NAMES`).

## Tests

```bash
python -m pytest
```
Couvrent le nettoyage, les features, le TF-IDF, la classification (dont l'absence de fuite), la validation, le clustering,
la recommandation, le pipeline et le DAG, la prédiction et l'exécution des pages Streamlit (avec un faux `streamlit`).
Le test des pages est ignoré tant que le pipeline n'a pas produit les fichiers.

## Structure

```
src/        extraction, cleaning, database, features, nlp, models, recommendation, pipeline.py
airflow/    dags/movie_pipeline_dag.py
app/        streamlit_app.py, shared.py, pages/
tests/      tests unitaires
data/       raw/ (JSON TMDB), processed/, features/
models/     objets joblib (vectoriseur, classifieur, K-Means)
reports/    figures/ et rapports générés
notebooks/  exploration pas à pas (jours 2 à 5)
```

## Limites connues

- Catalogue trié par popularité à l'extraction : biais vers les films récents et populaires.
- Les films récents ont peu de votes : un faible `vote_count` peut refléter la récence, pas un manque d'intérêt.
- Le cache d'extraction ne récupère pas de nouveaux films si `N_MOVIES` n'augmente pas (supprimer `data/raw/discover/` pour rafraîchir).
- La recommandation repose sur le texte seul : elle capte les thèmes, pas la qualité ni le ton.
- Airflow en mode `standalone` sans authentification : développement local uniquement.

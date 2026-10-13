# Movie Intelligence

Analyse, classification, clustering et recommandation de films à partir de l'API publique TMDB.

Le projet couvre toute la chaîne d'un projet de data science : extraction automatisée, nettoyage, stockage MongoDB,
analyse exploratoire, feature engineering, NLP (TF-IDF), classification, clustering, recommandation, orchestration avec
Airflow, conteneurisation avec Docker et application Streamlit.



## 1. Présentation

Une plateforme de contenus veut mieux comprendre son catalogue de films, identifier des profils de films et proposer des
recommandations. Les données (2 000 films) sont extraites de TMDB et combinent des variables structurées (notes, durée,
date, genres, budget, revenus) et textuelles (titres, synopsis, mots-clés).

Le projet répond à trois problématiques :

| Problématique | Question | Méthode |
|---|---|---|
| **Classification** | Un film aura-t-il un fort engagement du public ? | Pipeline scikit-learn, 3 modèles comparés, validation croisée, GridSearchCV |
| **Clustering** | Quels profils de films existe-t-il ? | K-Means sur variables structurées et sur le texte (TF-IDF + SVD), Silhouette Score |
| **Recommandation** | Quels films ressemblent à celui-ci ? | TF-IDF et similarité cosinus |

La cible de la classification est `high_engagement` : un film a un fort engagement s'il a reçu au moins **4 000 votes** sur TMDB.

---

## 2. Architecture

```
API TMDB
   │
   ▼
extract ──► data/raw/ (JSON bruts, cache local)
   │
   ▼
clean ────► data/processed/movies_clean.csv
   │
   ▼
features ─► data/features/movies_features.csv
   │
   ▼
mongo ────► MongoDB (movie_db.movies)
   │
   ▼
tfidf ────► models/tfidf_vectorizer.joblib + data/features/tfidf_matrix.npz
   │
   ▼
classifier ► models/best_classifier.joblib
   │
   ▼
clustering ► models/kmeans_*.joblib + data/features/movie_clusters.csv
   │
   ▼
Application Streamlit (charge les fichiers .joblib, sans réentraîner)
```

Les sept étapes sont orchestrées par un DAG Airflow (`movie_pipeline`) et peuvent aussi être lancées à la main.

### Services Docker

| Service | Rôle | Adresse |
|---|---|---|
| `mongo` | Base MongoDB | `localhost:27017` |
| `airflow` | Orchestration du pipeline (mode `standalone`) | http://localhost:8080 |
| `streamlit` | Application web | http://localhost:8501 |

Les services `airflow` et `streamlit` partagent la **même image Docker**. Les modèles `.joblib` dépendent de la version de
scikit-learn qui les a créés : les produire et les lire dans la même image évite toute incompatibilité.

---

## 3. Prérequis

- **Docker** avec Docker Compose v2 (commande `docker compose`)
- Un **compte TMDB** et son *API Read Access Token* (themoviedb.org > Paramètres > API)
- Environ 3 Go d'espace disque pour l'image Docker, et 4 Go de mémoire vive pour Airflow
- Sans Docker : Python 3.10 ou plus, et un MongoDB accessible

---

## 4. Lancement avec Docker (recommandé)

À la racine du projet :

```bash
cp .env.example .env                       # puis renseigner TMDB_READ_TOKEN dans .env
echo "AIRFLOW_UID=$(id -u)" >> .env        # une seule fois (voir la section Dépannage)
docker compose up -d --build               # première fois : plusieurs minutes
```

Lancer le pipeline :

```bash
docker compose exec airflow airflow dags unpause movie_pipeline
docker compose exec airflow airflow dags trigger movie_pipeline
```

Puis :

1. Ouvrir **http://localhost:8080** (aucun mot de passe en local) : les 7 tâches passent au vert.
2. Une fois le pipeline terminé, ouvrir **http://localhost:8501** : l'application Streamlit.

Commandes utiles :

```bash
docker compose ps                          # état des conteneurs
docker compose logs -f airflow             # journaux d'Airflow
docker compose logs streamlit              # journaux de l'application
docker compose restart streamlit           # après une modification du code de l'application
docker compose stop airflow                # empêche le lancement hebdomadaire automatique
docker compose down                        # arrêt complet (les données restent dans le dossier du projet)
```

Le projet est monté dans les conteneurs : les fichiers produits (`data/`, `models/`, `reports/`) apparaissent directement
dans votre dossier et survivent à l'arrêt des conteneurs. Une modification du code Python est prise en compte sans
reconstruire l'image ; seule une modification de `requirements.txt` ou du `Dockerfile` impose `docker compose up -d --build`.

---

## 5. Lancement sans Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # renseigner TMDB_READ_TOKEN
docker compose up -d mongo                 # MongoDB seul (nécessaire pour l'étape mongo)

python -m src.pipeline                     # les 7 étapes dans l'ordre
python -m src.pipeline clean features      # ou seulement certaines étapes
streamlit run app/streamlit_app.py         # application sur http://localhost:8501
```

Étapes disponibles : `extract`, `clean`, `features`, `mongo`, `tfidf`, `classifier`, `clustering`.

Scripts indépendants du pipeline :

```bash
python src/extraction/extract_tmdb.py --n-movies 40     # test rapide de l'extraction
python src/extraction/check_raw.py                      # contrôle qualité des JSON bruts
python src/eda/eda_plots.py                             # 11 graphiques d'EDA + interprétations
python src/models/classification.py                     # 3 modèles, baseline, démonstration de Data Leakage
python src/database/mongo_queries.py                    # requêtes et agrégations MongoDB
python src/recommendation/recommender.py --title "Inception" --n 5
python src/recommendation/recommender.py --evaluate     # qualité globale de la recommandation
```

Attention : les modèles doivent être lus avec la même version de scikit-learn que celle qui les a créés. Il faut donc soit
tout faire dans Docker, soit tout faire hors Docker, mais ne pas mélanger les deux.

---

## 6. Configuration

Fichier `.env` à la racine (modèle : `.env.example`). Il ne doit jamais être versionné.

| Variable | Rôle | Défaut |
|---|---|---|
| `TMDB_READ_TOKEN` | Jeton d'accès TMDB (v4), prioritaire | (obligatoire) |
| `TMDB_API_KEY` | Clé TMDB (v3), utilisée si le jeton est absent | |
| `MONGO_URI` | Adresse de MongoDB | `mongodb://localhost:27017` (dans Docker : `mongodb://mongo:27017`, imposé par le compose) |
| `MONGO_DB` | Nom de la base | `movie_db` |
| `MONGO_COLLECTION` | Nom de la collection | `movies` |
| `AIRFLOW_UID` | Identifiant utilisateur des conteneurs (valeur de `id -u`) | `50000` |
| `N_MOVIES` | Nombre de films à extraire | `2000` |
| `TFIDF_KEYWORDS` | `1` : vectoriser synopsis et mots-clés TMDB ; `0` : synopsis seul | `1` |

---

## 7. Le pipeline étape par étape

DAG `movie_pipeline` : `extract → clean → features → mongo → tfidf → classifier → clustering`.
Planification hebdomadaire (`@weekly`), une nouvelle tentative automatique par tâche après 2 minutes, une seule exécution
à la fois. Le DAG est créé en pause : il faut l'activer (`unpause`).

### extract : `src/extraction/extract_tmdb.py`

Deux phases : `GET /discover/movie` (liste paginée d'identifiants, triée par popularité, au moins 10 votes) puis
`GET /movie/{id}?append_to_response=keywords` (détails et mots-clés en un seul appel).
Authentification par jeton Bearer lu dans `.env`, nouvelles tentatives avec attente croissante sur les erreurs 429 et 5xx,
pause entre les appels, gestion des réponses vides et des 404. Chaque réponse est sauvegardée en JSON dans `data/raw/` :
un fichier déjà présent n'est jamais retéléchargé (cache local).
L'étape échoue si aucun film n'est téléchargé ou si plus de 10 % des appels échouent.

### clean : `src/cleaning/clean_data.py`

- Aplatit les JSON imbriqués (genres et mots-clés deviennent des listes de noms).
- Analyse : types, valeurs manquantes, doublons, incohérences.
- Corrections : doublons supprimés (`movie_id`, puis titre et date), dates converties, `budget` et `revenue` inférieurs à
  1 000 $ mis à `NaN` (0 signifie « inconnu » chez TMDB), `runtime` à 0 mis à `NaN`, note sans vote mise à `NaN`,
  films non sortis supprimés, texte vide mis à `NaN`.
- **Aucune imputation** à ce stade (voir la section 8).
- Sorties : `data/processed/movies_clean.csv`, `column_types.json` (variables numériques, catégorielles, textuelles,
  dates, listes), `reports/cleaning_report.txt`.

### features : `src/features/build_features.py`

Variables créées, toutes calculées ligne par ligne : année, mois (aussi en sinus et cosinus), trimestre, décennie,
`n_genres`, `n_keywords`, catégorie de durée (seuils fixes 90, 120, 150 minutes), `has_budget`, `budget_log`,
`is_english`, longueurs du synopsis et du titre, one-hot des genres.
Sorties : `data/features/movies_features.csv`, `feature_groups.json` (groupes pour le `ColumnTransformer`),
`reports/feature_justifications.md` (justification de chaque variable et liste des variables écartées).

### mongo : `src/database/`

`mongo_loader.py` charge les films dans MongoDB par *upsert* (`_id = movie_id`) : relancer l'étape ne crée aucun doublon.
Des index sont créés sur les genres, la langue, la date, la note et la popularité.
`mongo_queries.py` contient 7 requêtes (tri, filtres composés, filtre sur tableau, `$all`, expression régulière,
`count_documents`, `distinct`) et 4 agrégations (`$match`, `$unwind`, `$group`, `$sort`, `$project`, `$limit`) :
bilan par genre, par décennie, par langue et mots-clés les plus fréquents.

### tfidf : `src/nlp/tfidf.py`

Nettoyage du texte (minuscules, suppression du HTML, des URL, des chiffres et de la ponctuation), valeurs manquantes
remplacées par une chaîne vide, `TfidfVectorizer` (`min_df=2`, `max_df=0.8`, normalisation L2).
Quinze combinaisons `max_features` × `ngram_range` sont testées et comparées avec une mesure sans étiquette : la
**cohérence des genres** (chevauchement de genres entre un film et ses 5 plus proches voisins, comparé au hasard).
À cohérence égale, la combinaison au plus petit vocabulaire est retenue.
Sorties : `models/tfidf_vectorizer.joblib`, `data/features/tfidf_matrix.npz`, `movies_text.csv` (synopsis nettoyés),
`reports/tfidf_experiments.csv`, figures 12 à 14.

### classifier : `src/models/classification.py` et `validation.py`

- Cible : `high_engagement = (vote_count >= 4000)`.
- Pipeline : `ColumnTransformer` (imputation par la médiane et standardisation des numériques, one-hot des catégorielles,
  TF-IDF du synopsis) suivi d'un modèle.
- Trois modèles comparés : régression logistique, Random Forest, SVM linéaire, plus une baseline (classe majoritaire).
- Métriques : accuracy, precision, recall, F1, ROC-AUC, matrices de confusion.
- `validation.py` (étape du DAG) : `StratifiedKFold` à 5 plis avec `cross_validate`, puis `GridSearchCV` sur le modèle ayant
  le meilleur ROC-AUC en validation croisée, avec comparaison avant et après optimisation sur le jeu de test.
- Sorties : `models/best_classifier.joblib`, `best_params.json`, `reports/cv_results.csv`, `gridsearch_results.csv`,
  `before_after.csv`, `classification_results.csv`, `classification_leakage_demo.csv`, figures 15 à 18.

### clustering : `src/models/clustering.py`

- **Variables structurées** : année, durée, budget (log), note, nombre de votes (log), nombre de genres ; imputation par la
  médiane et standardisation.
- **Texte** : TF-IDF réduit à 50 dimensions par `TruncatedSVD`, puis normalisation.
- K de 2 à 10, Silhouette Score, K retenu = meilleur silhouette parmi K ≥ 3.
- Interprétation : tableaux de profils (nombre de films, année, durée, budget, note, votes, genres, mots caractéristiques),
  projections PCA et SVD en 2D.
- Sorties : `models/kmeans_structured.joblib`, `kmeans_text.joblib`, `data/features/movie_clusters.csv`,
  `reports/cluster_profiles_structured.csv`, `cluster_profiles_text.csv`, figures 19 et 20.

### Recommandation : `src/recommendation/recommender.py`

Sélection d'un film, récupération de son vecteur TF-IDF, similarité cosinus avec tout le catalogue, retour des 5 films les
plus proches avec titre, genres, année et score de similarité. Fonctionne aussi à partir d'une description libre.

---

## 8. Choix méthodologiques

**Éviter le Data Leakage**

- `vote_count` (qui définit la cible) n'est jamais une feature. `popularity`, `vote_average` et `revenue`, qui sont des
  proxys de l'engagement, sont aussi écartées. Une fonction (`assert_no_leakage`) lève une erreur si l'une d'elles se glisse
  dans les features, et `classification_leakage_demo.csv` montre l'effet de leur ajout.
- Imputation, standardisation, encodage et TF-IDF sont **dans le Pipeline** : ils sont ajustés sur le jeu d'entraînement
  uniquement. Le nettoyage du texte, sans état, est fait avant le découpage.
- Aucune feature ne dépend d'une statistique globale (seuils fixes pour les catégories de durée, pas de quantiles).
- Le seuil de la cible est fixé avant le découpage, et non recalculé sur le train ou le test.

**Validation**

- Le jeu de test est mis de côté ; la validation croisée et le GridSearch se font sur le jeu d'entraînement uniquement.
- Quand l'optimisation n'apporte pas de gain significatif (cas du Random Forest, où les paramètres par défaut sont
  retenus), le résultat est présenté tel quel.

**Texte et recommandation**

- Mots vides : liste anglaise, plus des mots génériques (`film`, `movie`, `story`), des métadonnées TMDB
  (`duringcreditsstinger`, `aftercreditsstinger`) et les noms de genres. Ces derniers sont redondants avec la colonne
  `genres` et rendraient l'évaluation circulaire.
- Les mots-clés TMDB sont ajoutés au texte vectorisé : gain net de cohérence (voir la section 9).

**Clustering**

- `vote_count` est utilisée dans le clustering structuré : sans cible à prédire, il n'y a pas de fuite, mais la part de
  films à fort engagement par cluster est donc descriptive, pas une validation indépendante.

---

## 9. Résultats

Résultats obtenus lors des exécutions du projet sur 2 000 films. Ils sont à mettre à jour si vous relancez le pipeline avec
d'autres données.

**Données**

| Indicateur | Valeur |
|---|---|
| Films extraits | 2 000 |
| `budget = 0` / `revenue = 0` | 14,1 % / 13,4 % |
| Films sans mots-clés | 1,8 % |
| Votes : médiane, 75e et 90e centiles | 4 120 / 7 712 / 13 482 |

**Classification** (validation croisée à 5 plis sur le jeu d'entraînement)

| Modèle | ROC-AUC | F1 |
|---|---|---|
| Random Forest | 0,869 ± 0,021 | 0,800 |
| Régression logistique | 0,823 ± 0,014 | 0,764 |
| SVM linéaire | 0,787 ± 0,012 | 0,736 |

Modèle retenu : **Random Forest**. Le GridSearch confirme les paramètres par défaut. Sur le jeu de test : ROC-AUC 0,881,
F1 0,821, accuracy 0,81. Pour la régression logistique, l'optimisation n'apporte pas de gain significatif
(AUC de test 0,798 avant, 0,803 après).

**TF-IDF et recommandation** (cohérence des genres des 5 recommandations ; hasard : 0,153)

| Texte vectorisé | Cohérence | Rapport au hasard |
|---|---|---|
| Synopsis seul | 0,311 | × 2,0 |
| Synopsis et mots-clés (noms de genres retirés) | **0,416** | **× 2,7** |

Configuration retenue : 5 000 termes, unigrammes ; matrice 2 000 × 5 000, creuse à 99,3 %, environ 36 termes par film.

**Clustering structuré** : silhouette 0,198 pour K = 3 (structure faible : les groupes se chevauchent mais ont des profils
lisibles).

| Cluster | Films | Profil |
|---|---|---|
| 0 | 422 | Classiques bien notés (années 1990, note moyenne 7,5) |
| 1 | 564 | Sorties récentes à faible audience (2024, 8 % à fort engagement) |
| 2 | 1 014 | Blockbusters récents multi-genres (budget médian 90 M$) |

Le clustering du texte donne un silhouette très faible (inférieur à 0,1) : les thèmes forment un continuum plutôt que des
groupes nets ; les clusters restent interprétables par leurs mots caractéristiques (policier, lycée, guerre, romance…).

**Pipeline Airflow** : 7 tâches en succès, environ 45 secondes avec le cache d'extraction.

---

## 10. Application Streamlit

| Page | Contenu |
|---|---|
| **Accueil** | Indicateurs, filtres (années, genres), distributions, genres, sorties par année, budget et revenus, votes et popularité, figures de l'EDA |
| **Classification** | Prédiction de l'engagement d'un film saisi (durée, année, budget, langue, genres, synopsis), performances des modèles |
| **Clusters** | Profils des clusters (variables structurées ou texte), exploration d'un cluster, cluster d'un synopsis |
| **Recommandation** | 5 films similaires (titre, genres, année, similarité) à partir d'un film ou d'une description |

L'application charge les fichiers `.joblib` et les données produits par le pipeline ; elle ne réentraîne rien. Si un fichier
manque, la page l'indique et demande de lancer le pipeline. Les noms des clusters se renseignent dans `app/shared.py`
(`CLUSTER_NAMES`).

---

## 11. Tests

```bash
python -m pytest
```

47 tests : nettoyage, features, TF-IDF, classification (dont l'absence de fuite et l'ajustement du vocabulaire sur le train
uniquement), validation, clustering, recommandation, pipeline, structure du DAG, extraction avec une fausse API,
prédiction pour un film saisi, et exécution des pages Streamlit avec un faux module `streamlit`. Le test des pages est ignoré
tant que le pipeline n'a pas produit les fichiers de données et de modèles.

---

## 12. Structure du dépôt

```
.
├── .env / .env.example / .gitignore / .dockerignore
├── Dockerfile                    image : Airflow 3.2.2 + dépendances du projet
├── docker-compose.yml            services mongo, airflow, streamlit
├── requirements.txt / pytest.ini
├── README.md
├── airflow/dags/
│   └── movie_pipeline_dag.py     DAG : chaîne linéaire de 7 tâches
├── app/
│   ├── streamlit_app.py          page d'accueil (tableau de bord)
│   ├── shared.py                 chargements avec cache, chemins, noms des clusters
│   └── pages/                    1_Classification, 2_Clusters, 3_Recommandation
├── src/
│   ├── pipeline.py               les 7 étapes, appelables par Airflow ou en ligne de commande
│   ├── extraction/               extract_tmdb.py, check_raw.py
│   ├── cleaning/                 clean_data.py
│   ├── database/                 mongo_loader.py, mongo_queries.py
│   ├── eda/                      eda_plots.py
│   ├── features/                 build_features.py
│   ├── nlp/                      tfidf.py
│   ├── models/                   classification.py, validation.py, clustering.py, predict.py
│   └── recommendation/           recommender.py
├── notebooks/                    exploration pas à pas (nettoyage, EDA, features, TF-IDF, classification)
├── tests/                        tests unitaires et de fumée
├── data/
│   ├── raw/                      JSON bruts TMDB (discover/ et movies/)
│   ├── processed/                données nettoyées
│   └── features/                 variables, matrice TF-IDF, clusters
├── models/                       objets joblib : vectoriseur, classifieur, K-Means
└── reports/                      rapports CSV et figures/ (20 graphiques numérotés)
```

Figures : 01 à 11 (EDA), 12 à 14 (TF-IDF), 15 à 17 (classification), 18 (validation croisée), 19 et 20 (clustering).

---

## 13. Correspondance avec le cahier des charges

| Étape du sujet | Réalisation |
|---|---|
| 1. Extraction et préparation | `src/extraction/extract_tmdb.py`, JSON dans `data/raw/` |
| 2. Nettoyage et stockage | `src/cleaning/clean_data.py`, `src/database/` (requêtes et agrégations MongoDB) |
| 3. EDA et visualisation | `src/eda/eda_plots.py` (11 graphiques, interprétations dans `reports/eda_interpretations.md`) |
| 4. Feature Engineering | `src/features/build_features.py` (justifications dans `reports/feature_justifications.md`) |
| 5. NLP avec TF-IDF | `src/nlp/tfidf.py` (`max_features`, `ngram_range`, dimensions, termes représentatifs) |
| 6. Classification | `src/models/classification.py` (3 modèles, Pipeline, ColumnTransformer, 6 métriques) |
| 7. Validation et optimisation | `src/models/validation.py` (StratifiedKFold, cross_validate, GridSearchCV, avant et après) |
| 8. Clustering | `src/models/clustering.py` (K-Means, Silhouette, TruncatedSVD 2D) |
| 9. Recommandation (bonus) | `src/recommendation/recommender.py` (cosinus, 5 films, titre, genres, année, score) |
| 10. Intégration | `app/` (Streamlit), `airflow/` (DAG), `Dockerfile` et `docker-compose.yml`, modèles sauvegardés avec joblib |

---

## 14. Dépannage

| Symptôme | Cause probable et solution |
|---|---|
| `network is unreachable` pendant `docker compose up --build` | Docker tente d'atteindre Docker Hub en IPv6 sans route. Relancer `docker pull apache/airflow:3.2.2` (les couches déjà reçues sont conservées) ou désactiver temporairement l'IPv6 : `sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1` puis `sudo systemctl restart docker` |
| `permission denied` sur `/opt/project` | Sous Fedora (SELinux), le volume doit porter `:z` (déjà présent dans le compose). Vérifier aussi que `AIRFLOW_UID` est défini dans `.env` |
| `No module named streamlit` | L'image a été construite avant l'ajout de `streamlit` dans `requirements.txt` : `docker compose up -d --build` |
| Un fichier modifié n'est pas pris en compte par l'application | Streamlit garde les objets en cache : `docker compose restart streamlit` |
| Le DAG n'apparaît pas | Attendre environ 30 secondes, puis `docker compose exec airflow airflow dags list-import-errors` |
| Une tâche est rouge, les suivantes orange | Les tâches suivantes sont annulées. Ouvrir la tâche rouge, onglet Logs, corriger, puis relancer avec `airflow dags trigger movie_pipeline` |
| L'étape `mongo` échoue | Vérifier que `movie_mongo` tourne (`docker compose ps`). Hors Docker, `MONGO_URI` doit pointer vers `localhost` |
| L'étape `extract` échoue | Jeton TMDB absent de `.env`, ou pas d'accès à Internet depuis le conteneur |
| Port 27017 déjà utilisé | Un ancien conteneur MongoDB tourne : `docker rm -f movie_mongo` |
| Avertissement de version scikit-learn au chargement d'un modèle | Modèle créé avec une autre version : relancer le pipeline dans l'environnement qui sert à le lire |
| Aucune nouvelle donnée après une nouvelle exécution | Le cache d'extraction ne retélécharge pas les films déjà présents : augmenter `N_MOVIES` ou supprimer `data/raw/discover/` |

---

## 15. Limites, sécurité et perspectives

**Limites**

- Le catalogue est trié par popularité à l'extraction : biais vers les films récents et populaires.
- Les films récents ont peu de votes : un faible `vote_count` peut refléter la récence plutôt qu'un manque d'intérêt.
- Le découpage train et test est aléatoire, non temporel.
- La recommandation repose sur le texte seul : elle capte les thèmes (un film sur le rêve recommande des films sur le rêve),
  pas le genre, le ton ni la qualité. La cohérence des genres est une mesure indirecte.
- Le clustering du texte est faiblement séparé (silhouette inférieure à 0,1).
- L'extraction est mise en cache : le pipeline hebdomadaire réentraîne surtout les modèles, il ne rafraîchit pas le
  catalogue tant que `N_MOVIES` n'augmente pas.

**Sécurité (usage local uniquement)**

- Airflow est lancé en mode `standalone`, sans authentification, et n'écoute que sur `127.0.0.1`.
- MongoDB n'a pas d'authentification et son port est publié sur toutes les interfaces dans `docker-compose.yml`
  (`"27017:27017"`). Hors poste personnel, restreindre à `"127.0.0.1:27017:27017"` et activer l'authentification.
- Le fichier `.env` contient le jeton TMDB : ne jamais le versionner.

**Perspectives**

- Lemmatisation ou plongements de phrases (embeddings) à la place de TF-IDF.
- Recommandation hybride : contenu, genre commun et popularité.
- Découpage temporel pour évaluer la classification.
- Airflow en production (base PostgreSQL, authentification, exécuteur distribué) et intégration continue des tests.

---

## 16. Crédits

Les données proviennent de TMDB. *This product uses the TMDB API but is not endorsed or certified by TMDB.*

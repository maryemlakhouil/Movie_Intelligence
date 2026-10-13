# Justification des variables créées

| Variable | Groupe | Justification |
|---|---|---|
| `release_year` | numeric | Époque de sortie  |
| `month_sin` | numeric | décembre est proche de janvier. |
| `month_cos` | numeric | Seconde composante du mois cyclique (sin et cos sont nécessaires ensemble). |
| `n_genres` | numeric | Un film multi-genres touche un public plus large. |
| `n_keywords` | numeric | Richesse de l'indexation : indicateur de la notoriété du film sur TMDB. |
| `runtime` | numeric | Durée brute en minutes. |
| `budget_log` | numeric | log(1+budget) : le budget est très asymétrique, le log rend la relation exploitable. |
| `overview_word_count` | numeric | Longueur du synopsis (mots) : les films documentés ont souvent un synopsis plus détaillé. |
| `overview_char_len` | numeric | Longueur du synopsis (caractères). |
| `title_word_count` | numeric | Longueur du titre en mots. |
| `has_budget` | binary | Budget connu ou non : l'absence d'information est elle-même un signal (les petits films sont moins documentés). |
| `is_english` | binary | Langue originale anglaise : marché et audience potentielle beaucoup plus larges. |
| `has_overview` | binary | Synopsis présent. |
| `genre_*` | binary | One-hot multi-label des genres (un film peut avoir plusieurs genres). Vocabulaire fermé TMDB : aucune fuite. |
| `runtime_category` | categorical | Court / moyen / long / très long (seuils fixes 90, 120, 150 min). |
| `original_language` | categorical | Langue originale (regroupement des langues rares fait dans le Pipeline). |
| `release_month` | descriptive_only | Mois de sortie (EDA). Le modèle utilise sa version cyclique sin/cos. |
| `release_quarter` | descriptive_only | Trimestre de sortie (EDA, saisonnalité). |
| `release_decade` | descriptive_only | Décennie (EDA, graphiques par période). Redondante avec release_year. |

## Variables écartées des features (risque de Data Leakage)

| Variable | Raison |
|---|---|
| `vote_count` | Sert à construire la cible high_engagement : interdit comme feature. |
| `vote_average` | Résultat direct de l'engagement du public (les votes) : proche de la cible. |
| `popularity` | Score TMDB calculé à partir des vues, votes et favoris : proxy de l'engagement. |
| `revenue` | Conséquence du succès du film, connue seulement après la sortie. |

# Rapport TF-IDF (Jour 4)

## 1. Qualité du texte

- Films : 2000
- Synopsis vides ou sans mot exploitable après nettoyage : 0 (0.0 %)
- Longueur médiane avant nettoyage : 42 mots ; après nettoyage et retrait des mots vides : 42 mots
- Synopsis très courts (moins de 10 mots nettoyés) : 1

## 2. Expériences max_features x ngram_range

min_df=2, max_df=0.8. La cohérence mesure le chevauchement de genres (Jaccard) entre un film et ses 5 voisins ; le hasard donne 0.153.

| ngram_min | ngram_max | max_features | vocab_size | nnz_per_doc | sparsity_pct | size_mb | genre_coherence | fit_seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 500 | 500 | 9.3 | 98.13 | 0.23 | 0.254 | 0.06 |
| 1 | 1 | 1000 | 1000 | 12.3 | 98.77 | 0.3 | 0.2743 | 0.06 |
| 1 | 1 | 2000 | 2000 | 15.6 | 99.22 | 0.38 | 0.2922 | 0.06 |
| 1 | 1 | 5000 | 5000 | 20.0 | 99.6 | 0.49 | 0.3111 | 0.06 |
| 1 | 1 |  | 5484 | 20.5 | 99.63 | 0.5 | 0.3145 | 0.06 |
| 1 | 2 | 500 | 500 | 9.4 | 98.11 | 0.23 | 0.2483 | 0.15 |
| 1 | 2 | 1000 | 1000 | 12.4 | 98.76 | 0.31 | 0.2715 | 0.15 |
| 1 | 2 | 2000 | 2000 | 15.9 | 99.2 | 0.39 | 0.2878 | 0.15 |
| 1 | 2 | 5000 | 5000 | 20.8 | 99.58 | 0.51 | 0.305 | 0.15 |
| 1 | 2 |  | 7536 | 23.3 | 99.69 | 0.57 | 0.3138 | 0.1 |
| 1 | 3 | 500 | 500 | 9.4 | 98.11 | 0.23 | 0.2492 | 0.26 |
| 1 | 3 | 1000 | 1000 | 12.5 | 98.75 | 0.31 | 0.2696 | 0.25 |
| 1 | 3 | 2000 | 2000 | 16.0 | 99.2 | 0.39 | 0.287 | 0.25 |
| 1 | 3 | 5000 | 5000 | 20.8 | 99.58 | 0.51 | 0.3042 | 0.25 |
| 1 | 3 |  | 7721 | 23.6 | 99.69 | 0.57 | 0.3139 | 0.14 |

**Lecture :**
- Unigrammes -> bigrammes : dégrade la cohérence (0.289 -> 0.285 en moyenne sur les max_features).
- Bigrammes -> trigrammes : ne change pratiquement pas la cohérence (0.285 -> 0.285).
- max_features de 500 à 5000 : améliore la cohérence (0.251 -> 0.307), au prix d'une matrice plus grande et d'un calcul plus long.

## 3. Configuration retenue et matrice finale

- `max_features=5000`, `ngram_range=(1, 1)`, `min_df=2`, `max_df=0.8`, `norm='l2'`
- Dimensions : **2000 films x 5000 termes**
- Valeurs non nulles : 39935 -> matrice creuse à **99.6 %** de zéros
- En moyenne 20.0 termes actifs par film sur 5000
- Mémoire (format creux) : 0.49 Mo ; en format dense elle ferait environ 80 Mo
- Vocabulaire : 5000 unigrammes et 0 n-grammes de plus d'un mot
- Cohérence des genres : 0.311 (hasard : 0.153, soit x2.0)

## 4. Termes représentatifs

### Les plus élevés en moyenne

`world`, `new`, `life`, `family`, `young`, `man`, `home`, `mysterious`, `love`, `finds`, `war`, `time`, `old`, `group`, `lives`, `years`, `friends`, `woman`, `help`, `father`

### Les plus caractéristiques par genre (écart à la moyenne globale)

- **Action** : mission, team, world, save, battle, enemy, ruthless, fight
- **Drama** : life, love, war, true, man, men, relationship, ii
- **Comedy** : school, new, friends, party, best, kids, adventure, trip
- **Adventure** : world, adventure, journey, bond, earth, peter, mission, quest
- **Thriller** : agent, police, ex, criminal, deadly, cia, killer, operative
- **Science Fiction** : earth, future, planet, alien, world, space, universe, threat

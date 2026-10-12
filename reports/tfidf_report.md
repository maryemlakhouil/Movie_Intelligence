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
| 1 | 1 | 500 | 500 | 17.2 | 96.55 | 0.42 | 0.3859 | 0.1 |
| 1 | 1 | 1000 | 1000 | 22.7 | 97.73 | 0.55 | 0.402 | 0.12 |
| 1 | 1 | 2000 | 2000 | 28.5 | 98.58 | 0.69 | 0.411 | 0.1 |
| 1 | 1 | 5000 | 5000 | 35.5 | 99.29 | 0.86 | 0.416 | 0.1 |
| 1 | 1 |  | 7377 | 38.0 | 99.48 | 0.92 | 0.4169 | 0.1 |
| 1 | 2 | 500 | 500 | 18.1 | 96.37 | 0.44 | 0.3844 | 0.48 |
| 1 | 2 | 1000 | 1000 | 24.2 | 97.58 | 0.59 | 0.4002 | 0.47 |
| 1 | 2 | 2000 | 2000 | 30.7 | 98.46 | 0.75 | 0.4077 | 0.45 |
| 1 | 2 | 5000 | 5000 | 39.6 | 99.21 | 0.96 | 0.4127 | 0.38 |
| 1 | 2 |  | 13565 | 49.8 | 99.63 | 1.2 | 0.4178 | 0.21 |
| 1 | 3 | 500 | 500 | 18.4 | 96.32 | 0.45 | 0.3818 | 0.8 |
| 1 | 3 | 1000 | 1000 | 24.6 | 97.54 | 0.6 | 0.3956 | 0.72 |
| 1 | 3 | 2000 | 2000 | 31.4 | 98.43 | 0.76 | 0.4066 | 0.68 |
| 1 | 3 | 5000 | 5000 | 40.5 | 99.19 | 0.98 | 0.4132 | 0.63 |
| 1 | 3 |  | 15508 | 53.2 | 99.66 | 1.28 | 0.4161 | 0.33 |

**Lecture :**
- Unigrammes -> bigrammes : ne change pratiquement pas la cohérence (0.406 -> 0.405 en moyenne sur les max_features).
- Bigrammes -> trigrammes : ne change pratiquement pas la cohérence (0.405 -> 0.403).
- max_features de 500 à 5000 : améliore la cohérence (0.384 -> 0.414), au prix d'une matrice plus grande et d'un calcul plus long.

## 3. Configuration retenue et matrice finale

- `max_features=5000`, `ngram_range=(1, 1)`, `min_df=2`, `max_df=0.8`, `norm='l2'`
- Dimensions : **2000 films x 5000 termes**
- Valeurs non nulles : 70921 -> matrice creuse à **99.29 %** de zéros
- En moyenne 35.5 termes actifs par film sur 5000
- Mémoire (format creux) : 0.86 Mo ; en format dense elle ferait environ 80 Mo
- Vocabulaire : 5000 unigrammes et 0 n-grammes de plus d'un mot
- Cohérence des genres : 0.416 (hasard : 0.153, soit x2.7)

## 4. Termes représentatifs

### Les plus élevés en moyenne

`based`, `new`, `family`, `world`, `relationship`, `life`, `novel`, `book`, `sequel`, `love`, `war`, `young`, `man`, `secret`, `death`, `school`, `woman`, `father`, `murder`, `alien`

### Les plus caractéristiques par genre (écart à la moyenne globale)

- **Action** : superhero, secret, comic, hero, mission, agent, martial, universe
- **Drama** : love, true, war, relationship, life, novel, biography, ii
- **Comedy** : school, cartoon, hilarious, buddy, christmas, villain, friendship, best
- **Adventure** : superhero, space, world, villain, based, universe, sequel, comic
- **Thriller** : murder, killer, police, secret, agent, survival, assassin, revenge
- **Science Fiction** : alien, space, future, superhero, robot, dystopia, earth, planet

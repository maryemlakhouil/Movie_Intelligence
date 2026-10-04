# Interprétations de l'EDA

Jeu de données : 2000 films. Textes générés à partir des chiffres réels ; à relire et reformuler.

## Valeurs manquantes

Colonnes avec plus de 10 % de manquants : budget (14 %), revenue (13 %). Ces valeurs sont des informations inconnues de TMDB, pas des zéros : elles seront imputées dans un Pipeline (sur le train uniquement) et signalées par l'indicateur `has_budget`.

## Distribution des notes

La note moyenne est 6.93 (médiane 6.95, écart-type 0.79). La distribution est à peu près symétrique et 82 % des films ont une note entre 6 et 8 : les notes extrêmes sont rares, la note seule discrimine peu les films.

## Distribution de la popularité

La popularité est très asymétrique vers la droite (longue queue de valeurs élevées) : médiane 16.0 contre un maximum de 646. Quelques films concentrent l'essentiel de la popularité, ce qui écrase le graphique linéaire. L'échelle logarithmique (à droite) rend la distribution lisible : une transformation log sera utile.

## Distribution du nombre de votes

Le nombre de votes est très asymétrique vers la droite (longue queue de valeurs élevées) : médiane 4120, 75e centile 7712, 90e centile 13482, maximum 41307. Cette distribution guidera le choix du seuil de `high_engagement` au Jour 5 : la médiane donne des classes équilibrées, un centile plus élevé isole les films vraiment à fort engagement (classes déséquilibrées).

## Films par genre

Les genres les plus présents sont Action (705 films, 35 % du catalogue), Drama (618 films, 31 % du catalogue), Comedy (580 films, 29 % du catalogue). Un film pouvant avoir plusieurs genres, la somme des barres dépasse le nombre de films (2000). Le genre le plus rare est Documentary (8 films) : les déséquilibres entre genres peuvent influencer les clusters et les modèles.

## Sorties par année

Le catalogue couvre de 1933 à 2026. L'année la plus représentée est 2026 (224 films) et 82 % des films sont sortis depuis 2000. Le catalogue est donc fortement orienté vers les films récents (effet du tri par popularité à l'extraction), biais à mentionner : les conclusions valent surtout pour le cinéma récent.

## Durée

La durée médiane est de 111 minutes, la moitié des films durant entre 99 et 126 min. La distribution est asymétrique vers la droite. Le boxplot signale 67 valeurs aberrantes (durées très courtes ou très longues, maximum 254 min) : ce sont de vrais films atypiques, on les conserve, et les seuils fixes de `runtime_category` limitent leur influence.

## Budget vs revenus

Sur les 1662 films dont budget et revenus sont connus (83 % du catalogue), la corrélation de Spearman est de 0.68 : plus le budget est élevé, plus les revenus tendent à l'être. 90 % des films sont au-dessus de la droite revenue = budget. Observez la dispersion verticale : à budget égal, les revenus varient fortement. Les films sans budget ou revenus connus sont exclus du graphique.

## Votes vs popularité

La corrélation de Spearman entre votes et popularité est 0.29 (faible). Les deux mesures capturent en partie la même chose (l'attention du public). Point crucial pour la classification : si la cible est construite à partir de `vote_count`, `popularity` en est un proxy et pourrait provoquer un Data Leakage : nous l'écartons des features.

## Boxplots

Note médiane par décennie : la plus élevée pour les 1950s (7.74), la plus basse pour les 2000s (6.83). Les films anciens encore présents dans le catalogue sont probablement des classiques : biais de survie à mentionner. Côté engagement, le genre avec le plus de votes en médiane est Science Fiction (≈ 6086 votes) et le moins engageant Thriller (≈ 3501 votes), soit un rapport de 1.7. Le genre semble donc lié à l'engagement et sera une feature utile.

## Corrélations

Les paires les plus corrélées sont : budget / revenue (0.68), revenue / vote_count (0.63), runtime / vote_count (0.35). On utilise Spearman car les variables sont très asymétriques (Pearson serait dominé par les valeurs extrêmes). Les variables fortement corrélées entre elles sont redondantes pour les modèles linéaires, et celles liées à `vote_count` sont des sources potentielles de leakage.

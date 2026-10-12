# Rapport de classification (Jour 5)

## 1. Cible

`high_engagement` = 1 si `vote_count` >= 4000. Seuil fixe, défini avant le split (pas recalculé sur le train/test).
- Films : 2000 | positifs : 51.2 % | train : 1600 | test : 400 (split stratifié)
- Features : 10 numériques, 22 binaires, 2 catégorielles + TF-IDF de `overview_clean`. `vote_count`, `popularity`, `vote_average`, `revenue` exclues.

## 2. Résultats sur le jeu de test

| model | accuracy | precision | recall | f1 | roc_auc | train_accuracy |
|---|---|---|---|---|---|---|
| Baseline (classe majoritaire) | 0.5125 | 0.5125 | 1.0 | 0.6777 | 0.5 | 0.5119 |
| Logistic Regression | 0.7225 | 0.7098 | 0.7756 | 0.7413 | 0.7982 | 0.8538 |
| Random Forest | 0.81 | 0.7945 | 0.8488 | 0.8208 | 0.8807 | 1.0 |
| Linear SVM | 0.685 | 0.682 | 0.722 | 0.7014 | 0.7575 | 0.995 |

**Lecture :**
- Meilleur modèle selon le ROC-AUC : **Random Forest** (AUC 0.881, F1 0.821, accuracy 0.810).
- La baseline (toujours la classe majoritaire) obtient 0.512 d'accuracy : le meilleur modèle gagne 29.8 points de pourcentage.
- **Logistic Regression** : accuracy train 0.854 contre test 0.723 (écart 0.13) : signe de surapprentissage.
- **Random Forest** : accuracy train 1.000 contre test 0.810 (écart 0.19) : signe de surapprentissage.
- **Linear SVM** : accuracy train 0.995 contre test 0.685 (écart 0.31) : signe de surapprentissage.
- Matrice de confusion de Random Forest : 174 vrais positifs, 150 vrais négatifs, 45 faux positifs, 31 faux négatifs. Il se trompe plus souvent en annonçant un fort engagement à tort (précision < rappel).
- Un seul découpage train/test donne une estimation bruitée : la validation croisée du Jour 6 la consolidera.

## 3. Démonstration de Data Leakage

Même pipeline, mais en ajoutant `popularity`, `vote_average` et `revenue` aux features :

| modèle | AUC sans proxys | AUC avec proxys | F1 sans proxys | F1 avec proxys | gain AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7982 | 0.8643 | 0.7413 | 0.7838 | 0.0661 |
| Random Forest | 0.8807 | 0.9281 | 0.8208 | 0.8565 | 0.0474 |
| Linear SVM | 0.7575 | 0.8397 | 0.7014 | 0.7925 | 0.0822 |

Gain moyen d'AUC : +0.065. Le saut est net : ces variables contiennent déjà l'information sur le nombre de votes, le score devient artificiellement bon et ne reflète plus la capacité à prédire l'engagement à partir du contenu du film. Elles sont donc exclues du modèle final.

## 4. Variables influentes

Voir `reports/figures/17_feature_importance.png` (coefficients de la régression logistique et importances du Random Forest).


"""
Jour 6 - Validation croisée (StratifiedKFold + cross_validate) et optimisation (GridSearchCV).
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.classification import (RANDOM_STATE, TARGET, TEST_SIZE, TEXT_COL, THRESHOLD,  
                                       assert_no_leakage, build_pipeline, evaluate,
                                       get_feature_groups, get_models, load_dataset)

REPORTS_DIR, MODELS_DIR = PROJECT_ROOT / "reports", PROJECT_ROOT / "models"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Grilles d'hyperparamètres 
GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.1, 1, 10],
        "prep__text__ngram_range": [(1, 1), (1, 2)],      
    },
    "Random Forest": {
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 3],
    },
    "Linear SVM": {
        "clf__C": [0.01, 0.1, 1],
    },
}


def main(threshold: int, model_name: str | None, n_folds: int) -> None:

    df = load_dataset(threshold)
    groups = get_feature_groups(df)
    cols = groups["numeric"] + groups["binary"] + groups["categorical"] + [TEXT_COL]
    assert_no_leakage(cols)
    X_train, X_test, y_train, y_test = train_test_split(df[cols], df[TARGET], test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    # --- 1. Validation croisée des 3 modèles (sur le train uniquement) ------
    rows, fold_auc = [], {}
    for name, model in get_models().items():
        res = cross_validate(build_pipeline(model, groups), X_train, y_train, cv=cv, scoring=SCORING)
        fold_auc[name] = res["test_roc_auc"]
        rows.append({"model": name, **{m: f"{res[f'test_{m}'].mean():.3f} ± {res[f'test_{m}'].std():.3f}"
                                       for m in SCORING}})
    cv_table = pd.DataFrame(rows)
    print(f"\n=== Validation croisée ({n_folds} plis, moyenne ± écart-type) ===")
    print(cv_table.to_string(index=False))

    # --- 2. GridSearchCV sur un modèle (par défaut : le meilleur en validation croisée) ---
    if model_name is None:
        model_name = max(fold_auc, key=lambda n: fold_auc[n].mean())
        print(f"\nModèle optimisé : {model_name} (meilleur ROC-AUC en validation croisée)")
    pipe = build_pipeline(get_models()[model_name], groups)
    grid = GridSearchCV(pipe, GRIDS[model_name], cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)                     
    print(f"\n=== GridSearchCV : {model_name} ===")
    print("Meilleurs paramètres :", grid.best_params_)

    # --- 3. Avant / après, sur le test --------------------------------------

    pipe.fit(X_train, y_train)               # version par défaut
    before, after = evaluate(pipe, X_test, y_test), evaluate(grid.best_estimator_, X_test, y_test)
    compare = pd.DataFrame({
        "version": ["Avant (paramètres par défaut)", "Après (GridSearchCV)"],
        "AUC validation croisée": [fold_auc[model_name].mean(), grid.best_score_],
        "AUC test": [before["roc_auc"], after["roc_auc"]],
        "F1 test": [before["f1"], after["f1"]],
        "Accuracy test": [before["accuracy"], after["accuracy"]],
    }).round(4)
    print("\n=== Avant / après optimisation ===")
    print(compare.to_string(index=False))

    # --- Sauvegardes --------------------------------------------------------

    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    cv_table.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    compare.to_csv(REPORTS_DIR / "before_after.csv", index=False)
    (pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
     .sort_values("rank_test_score").to_csv(REPORTS_DIR / "gridsearch_results.csv", index=False))
    joblib.dump(grid.best_estimator_, MODELS_DIR / "best_classifier.joblib", compress=3)
    (MODELS_DIR / "best_params.json").write_text(
        json.dumps({"model": model_name, "params": grid.best_params_, "threshold": threshold}, default=str, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=pd.DataFrame(fold_auc), color="#3b6ea5", ax=ax)
    ax.set(title=f"ROC-AUC par pli ({n_folds} plis)", ylabel="ROC-AUC")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "18_cv_roc_auc.png", dpi=150)
    print(f"\nModèle final : {MODELS_DIR / 'best_classifier.joblib'}")


if __name__ == "__main__":
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=THRESHOLD)
    ap.add_argument("--model", default=None, choices=list(GRIDS))   # None = meilleur modèle en CV
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    main(a.threshold, a.model, a.folds)

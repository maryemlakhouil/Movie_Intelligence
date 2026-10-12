"""
ROC-AUC    est un score de 0,5 (hasard) à 1 (parfait) qui indique à quel point le modèle classe correctement 

RÈGLES ANTI-LEAKAGE
-------------------
1. : la cible vient de vote_count, donc vote_count ne peut pas être une feature 
2. vote_average, popularity et revenue (proxys) indice qui remplace indirectement y 
3. Imputation, mise à l'échelle, encodage et TF-IDF sont DANS le Pipeline : ils sont ajustés
   sur le train uniquement (fit) puis appliqués au test (transform).
"""
import argparse
import json
import logging
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier #reference
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features.build_features import GROUPS_JSON, LEAKAGE_RISK, feature_groups, load_features  
from src.nlp.tfidf import TEXT_CSV, make_vectorizer  

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODEL_PATH = MODELS_DIR / "classifier_day5.joblib"
META_PATH = MODELS_DIR / "classification_meta.json"
RESULTS_CSV = REPORTS_DIR / "classification_results.csv"

TARGET, TARGET_SOURCE = "high_engagement", "vote_count"
# seuil fixe, proche de la médiane observée (4 120 votes)
THRESHOLD = 4000      
TEXT_COL = "overview_clean"
TEST_SIZE, RANDOM_STATE = 0.2, 42
PROXY_FEATURES = ["popularity_log", "vote_average", "revenue_log"]  

logger = logging.getLogger("classification")


# 1. Données et cible

def make_target(df: pd.DataFrame, threshold: int = THRESHOLD) -> pd.Series:
    return (df[TARGET_SOURCE] >= threshold).astype(int).rename(TARGET)


def assert_no_leakage(feature_cols, allow: bool = False) -> None:
    if allow:# verifie les colonnes (arreter)
        return
    interdit = set(LEAKAGE_RISK) | {TARGET, TARGET_SOURCE}
    bad = [c for c in feature_cols if c in interdit or any(c.startswith(f + "_") for f in interdit)]
    if bad:
        raise ValueError(f"Data Leakage : colonnes interdites parmi les features -> {bad}")

# dataset final utilisé pour la classification

def load_dataset(threshold: int = THRESHOLD) -> pd.DataFrame:
    df = load_features()
    text = pd.read_csv(TEXT_CSV)[["movie_id", TEXT_COL]]
    df = df.merge(text, on="movie_id", how="left")
    df[TEXT_COL] = df[TEXT_COL].fillna("")                 
    df = df.dropna(subset=[TARGET_SOURCE]).reset_index(drop=True)
    df[TARGET] = make_target(df, threshold)
    df["popularity_log"] = np.log1p(df["popularity"])
    df["revenue_log"] = np.log1p(df["revenue"])
    return df


def get_feature_groups(df: pd.DataFrame) -> dict:
    if GROUPS_JSON.exists():
        g = json.loads(GROUPS_JSON.read_text(encoding="utf-8"))
    else:  
        g = feature_groups(df)
    return {k: [c for c in g[k] if c in df.columns] for k in ("numeric", "binary", "categorical")}


# 2. Pipeline : ColumnTransformer + modèle

def build_preprocessor(groups: dict, max_features: int = 3000, ngram_range=(1, 2)) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), groups["numeric"]),
            ("bin", "passthrough", groups["binary"]),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]),   #fr,en,ja
             groups["categorical"]),
            ("text", make_vectorizer(max_features, ngram_range), TEXT_COL),
        ]
    )


def get_models() -> dict:
    return {                                                                # cls minoritaire
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE, max_iter=10000),
    }


def build_pipeline(model, groups: dict) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(groups)), ("clf", model)])

# 3. Évaluation 

def get_scores(pipe, X) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X) 


def evaluate(pipe, X, y) -> dict:
    pred = pipe.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, get_scores(pipe, X)),
    }


def run_models(X_train, X_test, y_train, y_test, groups: dict, with_baseline: bool = True):
    models = get_models()
    if with_baseline:
        models = {"Baseline (classe majoritaire)": DummyClassifier(strategy="most_frequent"), **models}
    rows, fitted, preds = [], {}, {} #model entrainee 
    for name, model in models.items():
        pipe = build_pipeline(model, groups)
        pipe.fit(X_train, y_train)
        res = evaluate(pipe, X_test, y_test) 
        res["train_accuracy"] = accuracy_score(y_train, pipe.predict(X_train))
        rows.append({"model": name, **res}) #** cle valeur 
        fitted[name], preds[name] = pipe, pipe.predict(X_test)
        logger.info("%-30s acc=%.3f f1=%.3f auc=%.3f", name, res["accuracy"], res["f1"], res["roc_auc"])
    return pd.DataFrame(rows).round(4), fitted, preds


# 4. Graphiques

def _finish(fig, name: str, show: bool) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.show() if show else plt.close(fig)

# affiche matrice de confusions : 

def plot_confusions(preds: dict, y_test, show=False) -> None:
    names = [n for n in preds if not n.startswith("Baseline")] 
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2))
    for ax, n in zip(np.atleast_1d(axes), names):
        ConfusionMatrixDisplay.from_predictions(y_test, preds[n], ax=ax, colorbar=False, cmap="Blues",display_labels=["faible", "fort"])
        ax.set_title(n)
        ax.grid(False)
    _finish(fig, "15_confusion_matrices", show)


def plot_roc(fitted: dict, X_test, y_test, show=False) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n, pipe in fitted.items():
        if n.startswith("Baseline"):
            continue
        RocCurveDisplay.from_predictions(y_test, get_scores(pipe, X_test), name=n, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="hasard")
    ax.set(title="Courbes ROC (jeu de test)")
    ax.legend(loc="lower right")
    _finish(fig, "16_roc_curves", show)

# num__runtime -> runtime
def _clean_name(n: str) -> str:
    for p in ("num__", "bin__", "cat__", "text__"):
        n = n.replace(p, "") if n.startswith(p) else n
    return n

# regression logistic variables influencent le plus la décision du modèle 
# ravel ->2d en un seul dimention 
def top_coefficients(pipe, n: int = 12) -> pd.Series:
    names = [_clean_name(x) for x in pipe.named_steps["prep"].get_feature_names_out()]
    coef = pd.Series(pipe.named_steps["clf"].coef_.ravel(), index=names)
    s = coef.sort_values()
    return pd.concat([s.head(n), s.tail(n)])

# random forest   

def top_importances(pipe, n: int = 15) -> pd.Series:
    names = [_clean_name(x) for x in pipe.named_steps["prep"].get_feature_names_out()]
    return pd.Series(pipe.named_steps["clf"].feature_importances_, index=names).sort_values().tail(n)

# 
def plot_importances(fitted: dict, show=False) -> None:
    lr, rf = fitted["Logistic Regression"], fitted["Random Forest"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    c = top_coefficients(lr)
    axes[0].barh(c.index, c.values, color=np.where(c.values > 0, "#3b6ea5", "#e08e45"))
    axes[0].set(title="Régression logistique : coefficients (bleu = fort engagement)", xlabel="coefficient")
    r = top_importances(rf)
    axes[1].barh(r.index, r.values, color="#3b6ea5")
    axes[1].set(title="Random Forest : importance des variables", xlabel="importance")
    _finish(fig, "17_feature_importance", show)


# 6. Orchestration

def run_classification(threshold: int = THRESHOLD, leakage_demo: bool = True, show: bool = False) -> dict:
    df = load_dataset(threshold)
    groups = get_feature_groups(df)
    feature_cols = groups["numeric"] + groups["binary"] + groups["categorical"] + [TEXT_COL]
    assert_no_leakage(feature_cols)

    X, y = df[feature_cols], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    logger.info("Train %d | Test %d | positifs %.1f %%", len(X_train), len(X_test), y.mean() * 100)

    results, fitted, preds = run_models(X_train, X_test, y_train, y_test, groups)

    leak = None
    if leakage_demo:
        g2 = {**groups, "numeric": groups["numeric"] + PROXY_FEATURES}
        cols2 = g2["numeric"] + groups["binary"] + groups["categorical"] + [TEXT_COL]
        assert_no_leakage(cols2, allow=True)  # volontaire : c'est la démonstration
        Xa, Xb, ya, yb = train_test_split(df[cols2], y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
        leak, _, _ = run_models(Xa, Xb, ya, yb, g2, with_baseline=False)

    plot_confusions(preds, y_test, show)
    plot_roc(fitted, X_test, y_test, show)
    plot_importances(fitted, show)

    models_only = results[~results["model"].str.startswith("Baseline")]
    best = models_only.loc[models_only["roc_auc"].idxmax(), "model"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], MODEL_PATH, compress=3)
    META_PATH.write_text(json.dumps({
        "target": TARGET, "target_source": TARGET_SOURCE, "threshold": threshold,
        "best_model": best, "feature_columns": feature_cols, "text_column": TEXT_COL,
        "test_metrics": models_only.set_index("model").loc[best, ["accuracy", "precision", "recall", "f1", "roc_auc"]].to_dict(),
    }, indent=2), encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)

    info = {"threshold": threshold, "n": len(df), "pos_pct": y.mean() * 100, "n_train": len(X_train),
            "n_test": len(X_test), "groups": groups, "results": results, "leak": leak,
            "preds": preds, "y_test": y_test, "fitted": fitted, "best": best,
            "X_test": X_test}
    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=THRESHOLD)
    ap.add_argument("--no-leakage-demo", action="store_true")
    a = ap.parse_args()
    out = run_classification(a.threshold, not a.no_leakage_demo)
    print("\n" + out["results"].to_string(index=False))
    if out["leak"] is not None:
        print("\n--- AVEC les proxys (démonstration de leakage) ---")
        print(out["leak"][["model", "accuracy", "f1", "roc_auc"]].to_string(index=False))
    print(f"\nMeilleur modèle : {out['best']}\nModèle : {MODEL_PATH}")
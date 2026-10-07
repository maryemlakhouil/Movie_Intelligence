"""
ROC-AUC    est un score de 0,5 (hasard) à 1 (parfait) 
           qui indique à quel point le modèle classe correctement 

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
REPORT_MD = REPORTS_DIR / "classification_report.md"

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


# --------------------------------------------------------------------------
# 3. Évaluation
# --------------------------------------------------------------------------
def get_scores(pipe, X) -> np.ndarray:
    """Score continu pour le ROC-AUC : probabilité si disponible, sinon distance à l'hyperplan (SVM)."""
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
    """Entraîne et évalue chaque modèle. Retourne (tableau de résultats, pipelines, prédictions)."""
    models = get_models()
    if with_baseline:
        models = {"Baseline (classe majoritaire)": DummyClassifier(strategy="most_frequent"), **models}
    rows, fitted, preds = [], {}, {}
    for name, model in models.items():
        pipe = build_pipeline(model, groups)
        pipe.fit(X_train, y_train)
        res = evaluate(pipe, X_test, y_test)
        res["train_accuracy"] = accuracy_score(y_train, pipe.predict(X_train))
        rows.append({"model": name, **res})
        fitted[name], preds[name] = pipe, pipe.predict(X_test)
        logger.info("%-30s acc=%.3f f1=%.3f auc=%.3f", name, res["accuracy"], res["f1"], res["roc_auc"])
    return pd.DataFrame(rows).round(4), fitted, preds


# --------------------------------------------------------------------------
# 4. Graphiques
# --------------------------------------------------------------------------
def _finish(fig, name: str, show: bool) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.show() if show else plt.close(fig)


def plot_confusions(preds: dict, y_test, show=False) -> None:
    names = [n for n in preds if not n.startswith("Baseline")]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2))
    for ax, n in zip(np.atleast_1d(axes), names):
        ConfusionMatrixDisplay.from_predictions(y_test, preds[n], ax=ax, colorbar=False, cmap="Blues",
                                                display_labels=["faible", "fort"])
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


def _clean_name(n: str) -> str:
    for p in ("num__", "bin__", "cat__", "text__"):
        n = n.replace(p, "") if n.startswith(p) else n
    return n


def top_coefficients(pipe, n: int = 12) -> pd.Series:
    names = [_clean_name(x) for x in pipe.named_steps["prep"].get_feature_names_out()]
    coef = pd.Series(pipe.named_steps["clf"].coef_.ravel(), index=names)
    s = coef.sort_values()
    return pd.concat([s.head(n), s.tail(n)])


def top_importances(pipe, n: int = 15) -> pd.Series:
    names = [_clean_name(x) for x in pipe.named_steps["prep"].get_feature_names_out()]
    return pd.Series(pipe.named_steps["clf"].feature_importances_, index=names).sort_values().tail(n)


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


# --------------------------------------------------------------------------
# 5. Rapport
# --------------------------------------------------------------------------
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def write_report(info: dict) -> None:
    res = info["results"]
    models = res[~res["model"].str.startswith("Baseline")].set_index("model")
    base = res[res["model"].str.startswith("Baseline")]
    best = models["roc_auc"].idxmax()
    L = ["# Rapport de classification (Jour 5)", "",
         "## 1. Cible", "",
         f"`{TARGET}` = 1 si `{TARGET_SOURCE}` >= {info['threshold']}. Seuil fixe, défini avant le split (pas recalculé sur le train/test).",
         f"- Films : {info['n']} | positifs : {info['pos_pct']:.1f} % | train : {info['n_train']} | test : {info['n_test']} (split stratifié)",
         f"- Features : {len(info['groups']['numeric'])} numériques, {len(info['groups']['binary'])} binaires, "
         f"{len(info['groups']['categorical'])} catégorielles + TF-IDF de `{TEXT_COL}`. `{TARGET_SOURCE}`, `popularity`, `vote_average`, `revenue` exclues.", "",
         "## 2. Résultats sur le jeu de test", "", md_table(res), ""]
    L += ["**Lecture :**",
          f"- Meilleur modèle selon le ROC-AUC : **{best}** (AUC {models.loc[best, 'roc_auc']:.3f}, F1 {models.loc[best, 'f1']:.3f}, accuracy {models.loc[best, 'accuracy']:.3f})."]
    if len(base):
        gain = models.loc[best, "accuracy"] - base["accuracy"].iloc[0]
        L.append(f"- La baseline (toujours la classe majoritaire) obtient {base['accuracy'].iloc[0]:.3f} d'accuracy : "
                 f"le meilleur modèle {'gagne' if gain > 0 else 'perd'} {abs(gain) * 100:.1f} points de pourcentage.")
    for n, r in models.iterrows():
        gap = r["train_accuracy"] - r["accuracy"]
        if gap > 0.10:
            L.append(f"- **{n}** : accuracy train {r['train_accuracy']:.3f} contre test {r['accuracy']:.3f} (écart {gap:.2f}) : signe de surapprentissage.")
    tn, fp, fn, tp = confusion_matrix(info["y_test"], info["preds"][best]).ravel()
    L.append(f"- Matrice de confusion de {best} : {tp} vrais positifs, {tn} vrais négatifs, {fp} faux positifs, {fn} faux négatifs. "
             + ("Il se trompe plus souvent en annonçant un fort engagement à tort (précision < rappel)."
                if models.loc[best, "precision"] < models.loc[best, "recall"] else
                "Il rate plus de films à fort engagement qu'il n'en annonce à tort (rappel < précision)."
                if models.loc[best, "recall"] < models.loc[best, "precision"] else "Précision et rappel sont équilibrés."))
    L += ["- Un seul découpage train/test donne une estimation bruitée : la validation croisée du Jour 6 la consolidera.", ""]

    if info.get("leak") is not None:
        lk = info["leak"].set_index("model")
        L += ["## 3. Démonstration de Data Leakage", "",
              "Même pipeline, mais en ajoutant `popularity`, `vote_average` et `revenue` aux features :", ""]
        cmp_ = pd.DataFrame({
            "modèle": models.index,
            "AUC sans proxys": models["roc_auc"].values,
            "AUC avec proxys": lk.loc[models.index, "roc_auc"].values,
            "F1 sans proxys": models["f1"].values,
            "F1 avec proxys": lk.loc[models.index, "f1"].values,
        })
        cmp_["gain AUC"] = (cmp_["AUC avec proxys"] - cmp_["AUC sans proxys"]).round(4)
        L += [md_table(cmp_.round(4)), ""]
        g = cmp_["gain AUC"].mean()
        L += [f"Gain moyen d'AUC : {g:+.3f}. " + (
              "Le saut est net : ces variables contiennent déjà l'information sur le nombre de votes, le score devient artificiellement bon "
              "et ne reflète plus la capacité à prédire l'engagement à partir du contenu du film. Elles sont donc exclues du modèle final."
              if g > 0.03 else
              "Le gain est faible sur ces données, mais ces variables restent conceptuellement des proxys de la cible (conséquences de l'engagement) et sont écartées par principe."), ""]
    L += ["## 4. Variables influentes", "", "Voir `reports/figures/17_feature_importance.png` (coefficients de la régression logistique et importances du Random Forest).", ""]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# 6. Orchestration
# --------------------------------------------------------------------------
def run_classification(threshold: int = THRESHOLD, leakage_demo: bool = True, show: bool = False) -> dict:
    df = load_dataset(threshold)
    groups = get_feature_groups(df)
    feature_cols = groups["numeric"] + groups["binary"] + groups["categorical"] + [TEXT_COL]
    assert_no_leakage(feature_cols)

    X, y = df[feature_cols], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
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
    write_report(info)
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
    print(f"\nMeilleur modèle : {out['best']}\nRapport : {REPORT_MD}\nModèle : {MODEL_PATH}")

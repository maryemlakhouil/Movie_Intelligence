import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Classification", layout="wide")

from shared import CLASSIFIER_PATH, FIG_DIR, MODELS_DIR, REPORTS, THRESHOLD, load_classifier, load_movies, require  # noqa: E402

st.title("Classification : un film aura-t-il un fort engagement ?")
st.caption(f"Fort engagement = au moins {THRESHOLD} votes TMDB. Le modèle ne voit ni le nombre de votes, ni la popularité, "
           "ni la note, ni les revenus (pour éviter le Data Leakage).")
require(CLASSIFIER_PATH)

tab_pred, tab_perf = st.tabs(["Prédire pour un film", "Performances des modèles"])

with tab_pred:
    df = load_movies()
    languages = df["original_language"].value_counts().index.tolist()[:15]
    all_genres = sorted({g for lst in df["genres"] for g in lst})
    with st.form("film"):
        left, right = st.columns(2)
        title = left.text_input("Titre", "My new movie")
        runtime = left.slider("Durée (minutes)", 40, 240, 110)
        year = left.number_input("Année de sortie", 1950, 2035, 2024)
        month = left.selectbox("Mois de sortie", list(range(1, 13)), index=5)
        budget = right.number_input("Budget en $ (0 = inconnu)", min_value=0, value=0, step=1_000_000)
        language = right.selectbox("Langue originale", languages)
        n_keywords = right.slider("Nombre de mots-clés TMDB", 0, 40, 5)
        genres = st.multiselect("Genres", all_genres, default=all_genres[:1])
        overview = st.text_area("Synopsis (en anglais)",
                                "A group of astronauts travels through a wormhole to find a new home for humanity.")
        submitted = st.form_submit_button("Prédire")

    if submitted:
        from src.models.predict import make_input_row, predict_engagement
        row = make_input_row(runtime, year, month, budget, language, genres, overview, n_keywords, title)
        label, proba = predict_engagement(load_classifier(), row)
        st.subheader("Résultat")
        if proba is not None:
            st.metric("Probabilité de fort engagement", f"{proba:.0%}")
            st.progress(proba)
        (st.success if label == 1 else st.info)(
            f"Fort engagement probable (≥ {THRESHOLD} votes)" if label == 1 else "Engagement plus faible probable")
        st.caption("Estimation d'un modèle entraîné sur environ 2 000 films : à lire comme une tendance, pas comme une certitude.")

with tab_perf:
    for name, title in [("cv_results.csv", "Validation croisée (5 plis, moyenne ± écart-type)"),
                        ("before_after.csv", "Avant / après optimisation (GridSearchCV)")]:
        path = REPORTS / name
        if path.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(path), hide_index=True)
    params = MODELS_DIR / "best_params.json"
    if params.exists():
        st.subheader("Modèle retenu")
        st.json(json.loads(params.read_text(encoding="utf-8")))
    for stem, caption in [("15_confusion_matrices", "Matrices de confusion"), ("16_roc_curves", "Courbes ROC"),
                          ("17_feature_importance", "Variables influentes"), ("18_cv_roc_auc", "ROC-AUC par pli")]:
        path = FIG_DIR / f"{stem}.png"
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")

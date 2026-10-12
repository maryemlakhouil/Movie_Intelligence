import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clusters", layout="wide")

from shared import (CLUSTER_NAMES, CLUSTERS_CSV, FIG_DIR, KMEANS_TEXT_PATH, REPORTS, load_clusters,  # noqa: E402
                    load_kmeans_text, load_movies, load_vectorizer, require)

st.title("Clusters : profils de films")
require(CLUSTERS_CSV, KMEANS_TEXT_PATH)

mode = st.radio("Clustering sur", ["Variables structurées", "Texte (TF-IDF)"], horizontal=True)
key, col = ("structured", "cluster_structured") if mode.startswith("Var") else ("text", "cluster_text")
df = load_movies().merge(load_clusters(), on="movie_id")
names = CLUSTER_NAMES[key]

profile_path = REPORTS / f"cluster_profiles_{key}.csv"
if profile_path.exists():
    profile = pd.read_csv(profile_path)
    if names:
        profile.insert(1, "nom", profile["cluster"].map(names).fillna(""))
    st.subheader("Profil de chaque cluster")
    st.dataframe(profile, hide_index=True)

st.subheader("Taille des clusters")
st.bar_chart(df[col].value_counts().sort_index())

for stem, caption in [("19_silhouette_scores", "Silhouette Score selon K"), ("20_clusters_2d", "Projection en 2D")]:
    path = FIG_DIR / f"{stem}.png"
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")

st.subheader("Explorer un cluster")
cluster = st.selectbox("Cluster", sorted(df[col].unique()), format_func=lambda c: f"{c} : {names[c]}" if c in names else str(c))
top = df[df[col] == cluster].nlargest(10, "vote_count")[["title", "release_year", "genres", "vote_average", "vote_count"]].copy()
top["genres"] = top["genres"].apply(", ".join)
st.caption("Les 10 films les plus votés de ce cluster")
st.dataframe(top, hide_index=True)

if key == "text":
    st.subheader("Dans quel cluster thématique tombe un synopsis ?")
    text = st.text_area("Synopsis (en anglais)", "A detective hunts a serial killer in a small town while the police cover up the case.")
    if st.button("Trouver le cluster"):
        from src.nlp.tfidf import clean_text
        X = load_vectorizer().transform([clean_text(text)])
        if X.nnz == 0:
            st.warning("Aucun mot connu dans ce texte.")
        else:
            c = int(load_kmeans_text().predict(X)[0])
            st.success(f"Cluster {c}" + (f" : {names[c]}" if c in names else ""))

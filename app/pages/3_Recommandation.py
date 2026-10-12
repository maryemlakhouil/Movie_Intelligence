import streamlit as st

st.set_page_config(page_title="Recommandation", page_icon="🍿", layout="wide")

from shared import FEATURES_DIR, MODELS_DIR, load_recommender, load_vectorizer, require  # noqa: E402

st.title("🍿 Recommandation de films similaires")
st.caption("Similarité cosinus entre les synopsis (et mots-clés) vectorisés par TF-IDF : on recommande des films au contenu proche.")
require(FEATURES_DIR / "tfidf_matrix.npz", MODELS_DIR / "tfidf_vectorizer.joblib")

rec = load_recommender()
n = st.slider("Nombre de recommandations", 3, 10, 5)
tab_film, tab_text = st.tabs(["À partir d'un film", "À partir d'une description"])

with tab_film:
    labels = (rec.df["title"].astype(str) + " (" + rec.df["release_year"].astype("Int64").astype(str) + ")").tolist()
    default = int(rec.df["vote_count"].idxmax())
    choice = st.selectbox("Choisissez un film (vous pouvez taper pour chercher)", labels, index=default)
    i = labels.index(choice)
    row = rec.df.loc[i]
    st.write(f"**Genres :** {', '.join(row['genres'])}")
    st.write(str(row["overview"]) if isinstance(row["overview"], str) else "(pas de synopsis)")
    try:
        result = rec.recommend_by_index(i, n)
        st.subheader("Films similaires")
        st.dataframe(result, hide_index=True)
        st.bar_chart(result.set_index("title")["similarity"])
    except ValueError as exc:
        st.warning(str(exc))

with tab_text:
    text = st.text_area("Décrivez le film que vous aimeriez voir (en anglais)",
                        "A team of astronauts explores space and discovers an alien civilization.")
    if st.button("Recommander"):
        try:
            st.dataframe(rec.recommend_from_text(text, load_vectorizer(), n), hide_index=True)
        except ValueError as exc:
            st.warning(str(exc))

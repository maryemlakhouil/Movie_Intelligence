import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Intelligence", page_icon="🎬", layout="wide")

from shared import FEATURES_CSV, FIG_DIR, REPORTS, THRESHOLD, histogram, load_movies, require  # noqa: E402

st.title("🎬 Movie Intelligence : tableau de bord")
st.caption("Catalogue TMDB : analyse, classification de l'engagement, clusters et recommandations (menu à gauche).")
require(FEATURES_CSV)
df = load_movies()

# --- Filtres
st.sidebar.header("Filtres")
y_min, y_max = int(df["release_year"].min()), int(df["release_year"].max())
years = st.sidebar.slider("Années de sortie", y_min, y_max, (y_min, y_max))
all_genres = sorted({g for lst in df["genres"] for g in lst})
chosen = st.sidebar.multiselect("Genres (au moins un)", all_genres)

mask = df["release_year"].between(*years)
if chosen:
    mask &= df["genres"].apply(lambda lst: any(g in lst for g in chosen))
d = df[mask]
if d.empty:
    st.warning("Aucun film ne correspond à ces filtres.")
    st.stop()

# --- Indicateurs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Films", f"{len(d):,}".replace(",", " "))
c2.metric("Note moyenne", f"{d['vote_average'].mean():.2f}")
c3.metric("Votes (médiane)", f"{d['vote_count'].median():,.0f}".replace(",", " "))
c4.metric(f"Fort engagement (≥ {THRESHOLD} votes)", f"{(d['vote_count'] >= THRESHOLD).mean():.0%}")

tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Genres et époques", "Budget et popularité", "Figures d'analyse (EDA)"])

with tab1:
    a, b = st.columns(2)
    a.subheader("Notes")
    a.bar_chart(histogram(d["vote_average"], 20))
    b.subheader("Durée (minutes)")
    b.bar_chart(histogram(d["runtime"], 30))
    st.subheader("Nombre de votes (échelle log10)")
    st.bar_chart(histogram(np.log10(d["vote_count"].clip(lower=1)), 30))
    st.caption("Les votes sont très asymétriques : sans le log10, quelques films écraseraient tout le graphique.")

with tab2:
    a, b = st.columns(2)
    a.subheader("Films par genre")
    a.bar_chart(d.explode("genres")["genres"].dropna().value_counts())
    b.subheader("Sorties par année")
    b.bar_chart(d["release_year"].dropna().astype(int).value_counts().sort_index())
    st.caption("Un film peut avoir plusieurs genres : la somme des barres dépasse le nombre de films.")

with tab3:
    bud = d.dropna(subset=["budget", "revenue"])
    a, b = st.columns(2)
    a.subheader("Budget et revenus (log10)")
    if len(bud) > 2:
        a.scatter_chart(pd.DataFrame({"log10 budget": np.log10(bud["budget"]), "log10 revenus": np.log10(bud["revenue"])}),
                        x="log10 budget", y="log10 revenus")
        a.caption(f"{len(bud)} films avec budget et revenus connus. Corrélation de Spearman : "
                  f"{bud['budget'].corr(bud['revenue'], method='spearman'):.2f}")
    else:
        a.info("Pas assez de films avec budget et revenus connus.")
    b.subheader("Votes et popularité (log10)")
    b.scatter_chart(pd.DataFrame({"log10 votes": np.log10(d["vote_count"].clip(lower=1)),
                                  "log10 popularité": np.log10(d["popularity"].clip(lower=0.01))}),
                    x="log10 votes", y="log10 popularité")
    b.caption(f"Corrélation de Spearman : {d['vote_count'].corr(d['popularity'], method='spearman'):.2f}")

with tab4:
    figs = [p for p in sorted(FIG_DIR.glob("*.png")) if p.name[:2].isdigit() and int(p.name[:2]) <= 11]
    if not figs:
        st.info("Aucune figure : lancez `python src/eda/eda_plots.py`.")
    for p in figs:
        st.image(str(p), caption=p.stem[3:].replace("_", " "), width="stretch")
    interp = REPORTS / "eda_interpretations.md"
    if interp.exists():
        with st.expander("Interprétations de chaque graphique"):
            st.markdown(interp.read_text(encoding="utf-8"))

"""Tests unitaires du NLP / TF-IDF (Jour 4). Lancer : python -m pytest"""
import numpy as np
import pandas as pd

from src.nlp.tfidf import (
    build_text, clean_text, genre_coherence, make_vectorizer, matrix_stats,
    random_baseline, select_best, top_terms_by_genre, top_terms_global, transform_query,
)

DOCS = [
    "A haunted house with a vengeful ghost and a demon",
    "A cursed demon haunts the old house at night",
    "Two friends plan a hilarious wedding full of pranks",
    "A funny wedding goes wrong for two awkward friends",
    "An alien spaceship lands near a lonely planet colony",
    "Robots and aliens fight on a distant planet",
]
GENRES = [["Horror"], ["Horror"], ["Comedy"], ["Comedy"], ["Science Fiction"], ["Science Fiction"]]


def fit(docs=DOCS, **kw):
    clean = pd.Series(docs).map(clean_text)
    vec = make_vectorizer(max_features=None, ngram_range=(1, 1), min_df=1, max_df=1.0, **kw)
    return vec, vec.fit_transform(clean).tocsr()


def test_clean_text_basics():
    assert clean_text("The <b>Man's</b> DON'T visit http://x.com in 2024!!") == "the man dont visit in"
    assert clean_text(None) == "" and clean_text(float("nan")) == "" and clean_text("   ") == ""


def test_build_text_handles_missing():
    df = pd.DataFrame({"overview": ["Hello", None], "keywords": [["a", "b"], []]})
    assert list(build_text(df)) == ["Hello", ""]
    assert build_text(df, include_keywords=True).iloc[0] == "Hello a b"


def test_matrix_is_normalized_and_sparse():
    vec, X = fit()
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    assert np.allclose(norms, 1.0)
    st = matrix_stats(X)
    assert st["n_docs"] == 6 and st["sparsity_pct"] > 50


def test_empty_text_gives_zero_vector_not_error():
    vec, _ = fit()
    q = transform_query(vec, "")
    assert q.nnz == 0


def test_stopwords_removed_and_terms_meaningful():
    vec, X = fit()
    terms = set(vec.get_feature_names_out())
    assert "the" not in terms and "demon" in terms
    assert "demon" in top_terms_by_genre(vec, X, GENRES, ["Horror"], n=5, min_docs=1)["Horror"]
    assert len(top_terms_global(vec, X, 5)) == 5


def test_coherence_beats_random_baseline():
    _, X = fit()
    assert genre_coherence(X, GENRES, k=1) > random_baseline(GENRES)


def test_select_best_prefers_smaller_vocab_on_ties():
    exp = pd.DataFrame({
        "ngram_min": [1, 1], "ngram_max": [1, 2], "max_features": pd.array([500, None], dtype="Int64"),
        "vocab_size": [500, 5000], "genre_coherence": [0.300, 0.302],
    })
    assert select_best(exp) == (500, (1, 1))

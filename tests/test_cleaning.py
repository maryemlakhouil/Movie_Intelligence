"""Tests unitaires du nettoyage (Jour 2). Lancer : python -m pytest"""
import numpy as np
import pandas as pd

from src.cleaning.clean_data import clean_movies, split_variable_types


def make_df():
    return pd.DataFrame(
        {
            "movie_id": [1, 1, 2, 3, 4],
            "title": ["A", "A", "B", "C", "D"],
            "overview": ["ok", "ok", "   ", "text", "text"],
            "release_date": ["2000-01-01", "2000-01-01", "bad-date", "2010-05-05", "2015-05-05"],
            "runtime": [100, 100, 0, 90, 95],
            "original_language": [" EN", " EN", "fr", "fr", "en"],
            "status": ["Released", "Released", "Released", "Rumored", "Released"],
            "genres": [["Drama"], ["Drama"], [], ["Action"], ["Comedy"]],
            "keywords": [["x"], ["x"], [], None, ["y"]],
            "budget": [0, 0, 5_000_000, 12, 1_000_000],
            "revenue": [0, 0, 100, 2_000_000, 0],
            "popularity": [1.0, 1.0, 2.0, 3.0, 4.0],
            "vote_average": [7.0, 7.0, 0.0, 6.0, 5.0],
            "vote_count": [100, 100, 0, 50, 20],
        }
    )


def test_duplicates_removed():
    clean, _ = clean_movies(make_df())
    assert clean["movie_id"].is_unique


def test_placeholder_zeros_become_nan():
    clean, _ = clean_movies(make_df())
    row = clean[clean.movie_id == 1].iloc[0]
    assert np.isnan(row["budget"]) and np.isnan(row["revenue"])


def test_runtime_zero_and_bad_date_become_missing():
    clean, _ = clean_movies(make_df())
    row = clean[clean.movie_id == 2].iloc[0]
    assert np.isnan(row["runtime"])
    assert pd.isna(row["release_date"])
    assert pd.isna(row["overview"])


def test_unreleased_dropped_and_language_normalized():
    clean, _ = clean_movies(make_df())
    assert 3 not in clean["movie_id"].values
    assert set(clean["original_language"]) <= {"en", "fr"}


def test_no_imputation_zero_vote_average_is_nan():
    clean, _ = clean_movies(make_df())
    assert np.isnan(clean[clean.movie_id == 2].iloc[0]["vote_average"])


def test_lists_never_nan_and_types_split():
    clean, _ = clean_movies(make_df())
    assert all(isinstance(x, list) for x in clean["keywords"])
    types = split_variable_types(clean)
    assert "budget" in types["numeric"] and "overview" in types["text"]

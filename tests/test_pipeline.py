"""Tests automatiques : python -m pytest"""
import numpy as np
import pandas as pd
import pytest

from src.metrics import best_threshold, precision_at_k, scores_to_labels
from src.preprocessing import clean_title, filter_frequent_tags, load_posts, prepare_dataset
from src.tfidf_model import build_pipeline


@pytest.fixture
def tiny_csv(tmp_path):
    rows = [
        (1, "python", 0, "How to sort a list in Python"),
        (1, "list", 1, "How to sort a list in Python"),
        (2, "java", 0, "Java NullPointerException"),
        (3, "python", 0, "Python dict comprehension"),
    ]
    path = tmp_path / "posts.csv"
    pd.DataFrame(rows, columns=["post_id", "tag_name", "tag_position", "title"]).to_csv(path, index=False)
    return path


def test_load_posts_groups_tags_by_question(tiny_csv):
    posts = load_posts(tiny_csv)
    assert len(posts) == 3
    assert posts.loc[posts.post_id == 1, "tags"].item() == ["python", "list"]


def test_filter_frequent_tags_drops_rare_tags_and_empty_posts(tiny_csv):
    posts, kept = filter_frequent_tags(load_posts(tiny_csv), min_count=2)
    assert kept == ["python"]
    assert set(posts.post_id) == {1, 3}


@pytest.mark.parametrize("raw, expected", [
    ("How to use C# with .NET?", "how to use c# with .net"),
    ("C++ vs Java!!", "c++ vs java"),
    ("asp.net-mvc routing... help", "asp.net-mvc routing help"),
])
def test_clean_title_keeps_technical_tokens(raw, expected):
    assert clean_title(raw) == expected


def test_split_is_disjoint_and_complete():
    train, val, test, mlb = prepare_dataset()
    ids = [set(d.post_id) for d in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    total = sum(len(d) for d in (train, val, test))
    assert abs(len(test) / total - 0.15) < 0.02
    assert len(mlb.classes_) > 20


def test_scores_to_labels_always_predicts_at_least_one_tag():
    scores = np.array([[0.1, 0.2, 0.05], [0.9, 0.8, 0.1]])
    preds = scores_to_labels(scores, threshold=0.5)
    assert preds.tolist() == [[0, 1, 0], [1, 1, 0]]


def test_threshold_and_precision_at_k():
    y = np.array([[1, 0], [0, 1]])
    scores = np.array([[0.7, 0.3], [0.4, 0.6]])
    assert precision_at_k(y, scores, 1) == 1.0
    assert 0.3 < best_threshold(y, scores) <= 0.6


def test_tfidf_pipeline_learns_simple_examples():
    titles = ["python list sort", "python dict", "java exception", "java class"] * 5
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]] * 5)
    model = build_pipeline(C=10).fit(titles, y)
    assert model.predict_proba(["sort a python list"])[0].argmax() == 0

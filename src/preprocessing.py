"""Préparation des données, partagée par les deux modèles (TF-IDF et BERT).

Étapes :
1. charger le CSV (une ligne = un couple question/tag) ;
2. regrouper par question -> un titre + une liste de tags ;
3. ne garder que les tags assez fréquents pour être appris ;
4. découper en train / validation / test de façon stratifiée (multi-label).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "stackoverflow_posts.csv"
MIN_TAG_COUNT = 20
SEED = 42


def load_posts(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Charge le CSV et renvoie une ligne par question : post_id, title, tags (liste)."""
    raw = pd.read_csv(path)
    raw = raw.dropna(subset=["title", "tag_name"]).sort_values(["post_id", "tag_position"])
    posts = (
        raw.groupby("post_id")
        .agg(title=("title", "first"), tags=("tag_name", lambda s: list(dict.fromkeys(s))))
        .reset_index()
    )
    return posts


def filter_frequent_tags(posts: pd.DataFrame, min_count: int = MIN_TAG_COUNT) -> tuple[pd.DataFrame, list[str]]:
    """Garde uniquement les tags vus au moins `min_count` fois.

    Les questions qui n'ont plus aucun tag après filtrage sont retirées.
    """
    counts = posts["tags"].explode().value_counts()
    kept = sorted(counts[counts >= min_count].index)
    kept_set = set(kept)
    out = posts.copy()
    out["tags"] = out["tags"].apply(lambda tags: [t for t in tags if t in kept_set])
    out = out[out["tags"].str.len() > 0].reset_index(drop=True)
    return out, kept


def clean_title(text: str) -> str:
    """Nettoyage léger pour TF-IDF : minuscules, on garde les caractères utiles
    aux noms techniques (c#, c++, .net, asp.net-mvc) et on retire le reste."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9#+.\-\s]", " ", text)
    text = re.sub(r"[.\-](?![a-z0-9#+])|(?<![a-z0-9])-", " ", text)  # ponctuation isolée (garde ".net")
    return re.sub(r"\s+", " ", text).strip()


def split_train_val_test(
    posts: pd.DataFrame, labels: np.ndarray, val_size: float = 0.15, test_size: float = 0.15, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpage 70 / 15 / 15 qui respecte la répartition des tags dans chaque partie."""
    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(first.split(posts, labels))

    rel_val = val_size / (1 - test_size)
    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    train_rel, val_rel = next(second.split(trainval_idx, labels[trainval_idx]))
    train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]

    return (
        posts.iloc[train_idx].reset_index(drop=True),
        posts.iloc[val_idx].reset_index(drop=True),
        posts.iloc[test_idx].reset_index(drop=True),
    )


def prepare_dataset(path: str | Path = DATA_PATH, min_count: int = MIN_TAG_COUNT, seed: int = SEED):
    """Pipeline complet : renvoie (train, val, test, mlb).

    `mlb` transforme une liste de tags en vecteur 0/1 (une colonne par tag).
    Les deux modèles appellent cette fonction : ils voient exactement le même découpage.
    """
    posts = load_posts(path)
    posts, kept = filter_frequent_tags(posts, min_count)
    mlb = MultiLabelBinarizer(classes=kept)
    labels = mlb.fit_transform(posts["tags"])
    train, val, test = split_train_val_test(posts, labels, seed=seed)
    return train, val, test, mlb

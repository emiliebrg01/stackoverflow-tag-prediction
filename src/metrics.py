"""Métriques d'évaluation et choix du seuil de décision, communs aux deux modèles.

Un modèle multi-label donne, pour chaque tag, un score entre 0 et 1.
On prédit le tag si le score dépasse un seuil ; ce seuil est réglé sur la validation.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score


def scores_to_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Applique le seuil, en garantissant au moins un tag par question
    (si aucun score ne dépasse le seuil, on prend le tag le plus probable)."""
    preds = (scores >= threshold).astype(int)
    empty = preds.sum(axis=1) == 0
    preds[empty, scores[empty].argmax(axis=1)] = 1
    return preds


def best_threshold(y_true: np.ndarray, scores: np.ndarray, grid=None) -> float:
    """Cherche le seuil qui maximise le F1 micro."""
    grid = np.arange(0.05, 0.95, 0.01) if grid is None else grid
    f1s = [f1_score(y_true, scores_to_labels(scores, t), average="micro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 1) -> float:
    """Part des k tags les mieux classés qui sont corrects."""
    topk = np.argsort(-scores, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, topk, axis=1)
    return float(hits.mean())


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Toutes les métriques du rapport, arrondies."""
    y_pred = scores_to_labels(scores, threshold)
    res = {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_at_1": precision_at_k(y_true, scores, 1),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "threshold": threshold,
    }
    return {k: round(float(v), 4) for k, v in res.items()}

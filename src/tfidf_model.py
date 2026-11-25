"""Modèle 1 — approche Data Science classique : TF-IDF + régression logistique.

Lancer l'entraînement :  python -m src.tfidf_model
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from src.metrics import best_threshold, evaluate
from src.preprocessing import clean_title, prepare_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tfidf_model.joblib"
RESULTS_PATH = ROOT / "results" / "tfidf_metrics.json"
C_GRID = [0.3, 1, 3, 10]  # testé aussi : LinearSVC et LR sans class_weight -> moins bons


def build_pipeline(C: float = 10.0) -> Pipeline:
    """Texte -> vecteur TF-IDF (mots + morceaux de caractères) -> un classifieur par tag."""
    features = FeatureUnion([
        # mots seuls et paires de mots ; le token_pattern garde "c#", "c++", ".net"
        ("words", TfidfVectorizer(preprocessor=clean_title, token_pattern=r"[^\s]+",
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        # morceaux de 2 à 5 caractères : robuste aux variantes (jquery / jquery-ui)
        ("chars", TfidfVectorizer(preprocessor=clean_title, analyzer="char_wb",
                                  ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    clf = OneVsRestClassifier(
        LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=1000)
    )
    return Pipeline([("tfidf", features), ("clf", clf)])


def train(verbose: bool = True) -> dict:
    train_df, val_df, test_df, mlb = prepare_dataset()
    y_train, y_val, y_test = (mlb.transform(d["tags"]) for d in (train_df, val_df, test_df))

    # 1) Recherche du meilleur C (force de régularisation) sur la validation
    best = None
    for C in C_GRID:
        model = build_pipeline(C).fit(train_df["title"], y_train)
        scores = model.predict_proba(val_df["title"])
        t = best_threshold(y_val, scores)
        f1 = evaluate(y_val, scores, t)["f1_micro"]
        if verbose:
            print(f"C={C:>5}  seuil={t:.2f}  F1 micro (val)={f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, t, model)
    _, C, threshold, _ = best

    # 2) Modèle final, entraîné sur train et chronométré
    start = time.perf_counter()
    model = build_pipeline(C).fit(train_df["title"], y_train)
    train_time = time.perf_counter() - start

    # 3) Note finale sur le test (jamais vu)
    start = time.perf_counter()
    test_scores = model.predict_proba(test_df["title"])
    infer_ms = (time.perf_counter() - start) / len(test_df) * 1000

    metrics = evaluate(y_test, test_scores, threshold)
    metrics.update({"best_C": C, "n_tags": len(mlb.classes_), "train_time_s": round(train_time, 1),
                    "inference_ms_per_question": round(infer_ms, 3)})

    MODEL_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "mlb": mlb, "threshold": threshold}, MODEL_PATH)
    metrics["model_size_mb"] = round(MODEL_PATH.stat().st_size / 1e6, 1)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2))
    if verbose:
        print("\nRésultats sur le test :", json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()

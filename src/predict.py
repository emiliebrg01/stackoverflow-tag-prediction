"""Prédire les tags d'un titre avec le modèle TF-IDF entraîné.

Exemple :  python -m src.predict "How to center a div with CSS?"
"""
from __future__ import annotations

import sys

import joblib

from src.metrics import scores_to_labels
from src.tfidf_model import MODEL_PATH


class TagPredictor:
    def __init__(self, path=MODEL_PATH):
        bundle = joblib.load(path)
        self.model, self.mlb, self.threshold = bundle["model"], bundle["mlb"], bundle["threshold"]

    def predict(self, title: str) -> list[tuple[str, float]]:
        """Renvoie la liste (tag, score) des tags prédits, du plus sûr au moins sûr."""
        scores = self.model.predict_proba([title])
        mask = scores_to_labels(scores, self.threshold)[0].astype(bool)
        tags = [(tag, round(float(s), 3)) for tag, s, keep in zip(self.mlb.classes_, scores[0], mask) if keep]
        return sorted(tags, key=lambda x: -x[1])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage : python -m src.predict "titre de la question"')
    predictor = TagPredictor()
    for title in sys.argv[1:]:
        print(f"{title}\n  -> {predictor.predict(title)}")

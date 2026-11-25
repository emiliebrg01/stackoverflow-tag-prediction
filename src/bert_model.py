"""Modèle 2 — fine-tuning de BERT pour la prédiction multi-label de tags.

Prévu pour tourner sur GPU (Google Colab). Voir notebooks/02_bert_finetuning.ipynb.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.metrics import best_threshold, evaluate, scores_to_labels

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BertConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 48        # les titres font ~15 tokens : 48 suffit largement
    batch_size: int = 32
    lr_bert: float = 3e-5       # petit pas pour ne pas "casser" ce que BERT sait déjà
    lr_head: float = 1e-3       # pas plus grand pour la couche de sortie, partie de zéro
    weight_decay: float = 0.01
    epochs: int = 15
    patience: int = 3           # arrêt si pas d'amélioration pendant 3 epochs
    warmup_ratio: float = 0.1
    seed: int = 42


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(tokenizer, titles, labels, cfg: BertConfig, shuffle: bool) -> DataLoader:
    enc = tokenizer(list(titles), padding="max_length", truncation=True,
                    max_length=cfg.max_length, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.float32))
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_scores(model, loader, device) -> np.ndarray:
    """Probabilités (0-1) pour chaque tag."""
    model.eval()
    out = []
    for ids, mask, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def train_bert(train_df, val_df, test_df, mlb, cfg: BertConfig = BertConfig(), save_dir=None, verbose=True) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device()
    n_labels = len(mlb.classes_)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=n_labels, problem_type="multi_label_classification",
        id2label=dict(enumerate(mlb.classes_)), label2id={t: i for i, t in enumerate(mlb.classes_)},
    ).to(device)

    y_train, y_val, y_test = (mlb.transform(d["tags"]) for d in (train_df, val_df, test_df))
    train_dl = make_loader(tokenizer, train_df["title"], y_train, cfg, shuffle=True)
    val_dl = make_loader(tokenizer, val_df["title"], y_val, cfg, shuffle=False)
    test_dl = make_loader(tokenizer, test_df["title"], y_test, cfg, shuffle=False)

    # Deux vitesses d'apprentissage : corps de BERT (prudent) / tête de classification (rapide)
    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    optimizer = torch.optim.AdamW([
        {"params": body, "lr": cfg.lr_bert},
        {"params": head, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)
    total_steps = len(train_dl) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for ids, mask, labels in train_dl:
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits.float(), labels.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        val_scores = predict_scores(model, val_dl, device)
        t = best_threshold(y_val, val_scores)
        val_f1 = evaluate(y_val, val_scores, t)["f1_micro"]
        history.append({"epoch": epoch, "train_loss": round(running / len(train_dl), 4), "val_f1_micro": val_f1})
        if verbose:
            print(f"Epoch {epoch:2d} | loss train={history[-1]['train_loss']:.4f} | F1 micro val={val_f1:.4f} (seuil {t:.2f})")

        if val_f1 > best_f1:
            best_f1, best_state, bad_epochs = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if verbose:
                    print("Arrêt anticipé (early stopping).")
                break
    train_time = time.perf_counter() - start

    model.load_state_dict(best_state)
    threshold = best_threshold(y_val, predict_scores(model, val_dl, device))
    t0 = time.perf_counter()
    test_scores = predict_scores(model, test_dl, device)
    infer_ms = (time.perf_counter() - t0) / len(test_df) * 1000

    metrics = evaluate(y_test, test_scores, threshold)
    metrics.update({"model_name": cfg.model_name, "n_tags": n_labels, "epochs_run": len(history),
                    "train_time_s": round(train_time, 1), "inference_ms_per_question": round(infer_ms, 3),
                    "device": device.type, "model_size_mb": round(sum(p.numel() for p in model.parameters()) * 4 / 1e6, 1)})

    if save_dir is not None:
        save_dir = Path(save_dir)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        (save_dir / "tagger_config.json").write_text(json.dumps({"threshold": threshold, **asdict(cfg)}, indent=2))

    return {"metrics": metrics, "history": history, "model": model, "tokenizer": tokenizer,
            "threshold": threshold, "test_scores": test_scores, "y_test": y_test}


class BertTagger:
    """Prédiction à partir d'un modèle sauvegardé (save_dir de train_bert)."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        cfg = json.loads((model_dir / "tagger_config.json").read_text())
        self.threshold, self.max_length = cfg["threshold"], cfg["max_length"]
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.no_grad()
    def predict(self, title: str) -> list[tuple[str, float]]:
        enc = self.tokenizer([title], truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        scores = torch.sigmoid(self.model(**enc).logits).cpu().numpy()
        mask = scores_to_labels(scores, self.threshold)[0].astype(bool)
        tags = [(t, round(float(s), 3)) for t, s, keep in zip(self.labels, scores[0], mask) if keep]
        return sorted(tags, key=lambda x: -x[1])

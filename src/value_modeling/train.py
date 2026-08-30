from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from value_modeling.features import CAT_COLUMNS, NUM_COLUMNS, build_spec, load_joined, transform
from value_modeling.model import OTTValueModel


def make_loader(tensors: dict[str, torch.Tensor], indices, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        tensors["cat"][indices],
        tensors["num"][indices],
        tensors["tokens"][indices],
        tensors["value"][indices],
        tensors["watch"][indices],
        tensors["retained"][indices],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(data_dir: str, model_path: str, metrics_path: str, epochs: int = 6, seed: int = 42) -> None:
    torch.manual_seed(seed)
    df = load_joined(data_dir)
    spec = build_spec(df)
    tensors = transform(df, spec)
    train_idx, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=seed, stratify=df["retained_7d"])

    model = OTTValueModel(
        cat_cardinalities=[len(spec.cat_maps[col]) for col in CAT_COLUMNS],
        n_numeric=len(NUM_COLUMNS),
        vocab_size=len(spec.token_map),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    train_loader = make_loader(tensors, train_idx, 256, True)
    val_loader = make_loader(tensors, val_idx, 512, False)

    for _ in range(epochs):
        model.train()
        for cat, num, tokens, value, watch, retained in train_loader:
            out = model(cat, num, tokens)
            loss = mse(out["value"], value) + 0.01 * mse(out["watch"], watch) + bce(out["retention_logit"], retained)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    losses = []
    with torch.no_grad():
        for cat, num, tokens, value, watch, retained in val_loader:
            out = model(cat, num, tokens)
            losses.append(
                {
                    "value_mse": float(mse(out["value"], value)),
                    "watch_mse": float(mse(out["watch"], watch)),
                    "retention_bce": float(bce(out["retention_logit"], retained)),
                }
            )
    metrics = {k: sum(row[k] for row in losses) / len(losses) for k in losses[0]}
    metrics["n_train"] = len(train_idx)
    metrics["n_validation"] = len(val_idx)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    with open(str(Path(model_path).with_suffix(".spec.pkl")), "wb") as f:
        pickle.dump(spec, f)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PyTorch OTT value model.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="results/value_model.pt")
    parser.add_argument("--metrics-path", default="results/train_metrics.json")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.model_path, args.metrics_path, args.epochs, args.seed)

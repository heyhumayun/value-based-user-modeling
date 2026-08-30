from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score

from value_modeling.features import CAT_COLUMNS, NUM_COLUMNS, load_joined, transform
from value_modeling.model import OTTValueModel


def evaluate(data_dir: str, model_path: str, output_path: str) -> None:
    spec_path = str(Path(model_path).with_suffix(".spec.pkl"))
    with open(spec_path, "rb") as f:
        spec = pickle.load(f)
    df = load_joined(data_dir)
    tensors = transform(df, spec)
    model = OTTValueModel(
        cat_cardinalities=[len(spec.cat_maps[col]) for col in CAT_COLUMNS],
        n_numeric=len(NUM_COLUMNS),
        vocab_size=len(spec.token_map),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        out = model(tensors["cat"], tensors["num"], tensors["tokens"])
        value_pred = out["value"].squeeze().numpy()
        watch_pred = out["watch"].squeeze().numpy()
        retention_prob = torch.sigmoid(out["retention_logit"]).squeeze().numpy()
    metrics = {
        "value_rmse": float(mean_squared_error(df.value_score, value_pred, squared=False)),
        "watch_minutes_mae": float(mean_absolute_error(df.watch_minutes, watch_pred)),
        "retention_accuracy": float(accuracy_score(df.retained_7d, retention_prob >= 0.5)),
        "retention_roc_auc": float(roc_auc_score(df.retained_7d, retention_prob)),
        "mean_predicted_value": float(np.mean(value_pred)),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained OTT value model.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="results/value_model.pt")
    parser.add_argument("--output-path", default="results/eval_metrics.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.data_dir, args.model_path, args.output_path)

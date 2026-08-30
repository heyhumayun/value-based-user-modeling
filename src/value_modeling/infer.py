from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from value_modeling.features import CAT_COLUMNS, NUM_COLUMNS, load_joined, transform
from value_modeling.model import OTTValueModel


def infer(data_dir: str, model_path: str, output_path: str, n: int = 25) -> None:
    with open(str(Path(model_path).with_suffix(".spec.pkl")), "rb") as f:
        spec = pickle.load(f)
    df = load_joined(data_dir).sample(n, random_state=7).reset_index(drop=True)
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
        df["predicted_value_score"] = out["value"].squeeze().numpy().round(3)
        df["predicted_watch_minutes"] = out["watch"].squeeze().numpy().round(2)
        df["predicted_retention_probability"] = torch.sigmoid(out["retention_logit"]).squeeze().numpy().round(3)
    cols = [
        "user_id",
        "title",
        "genre",
        "entry_surface",
        "value_score",
        "predicted_value_score",
        "retained_7d",
        "predicted_retention_probability",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample inference for the OTT value model.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="results/value_model.pt")
    parser.add_argument("--output-path", default="results/sample_predictions.csv")
    parser.add_argument("--n", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args.data_dir, args.model_path, args.output_path, args.n)

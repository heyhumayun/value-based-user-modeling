from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from value_modeling.features import CAT_COLUMNS, NUM_COLUMNS
from value_modeling.io import load_joined


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLUMNS),
            ("num", StandardScaler(), NUM_COLUMNS),
        ]
    )


def run_baselines(data_dir: str, output_path: str, seed: int = 42) -> None:
    df = load_joined(data_dir)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["retained_7d"])
    x_train = train_df[CAT_COLUMNS + NUM_COLUMNS]
    x_test = test_df[CAT_COLUMNS + NUM_COLUMNS]

    value_model = Pipeline([("prep", _preprocessor()), ("model", GradientBoostingRegressor(random_state=seed))])
    watch_model = Pipeline([("prep", _preprocessor()), ("model", GradientBoostingRegressor(random_state=seed + 1))])
    retention_model = Pipeline([("prep", _preprocessor()), ("model", GradientBoostingClassifier(random_state=seed + 2))])

    value_model.fit(x_train, train_df["value_score"])
    watch_model.fit(x_train, train_df["watch_minutes"])
    retention_model.fit(x_train, train_df["retained_7d"])

    value_pred = value_model.predict(x_test)
    watch_pred = watch_model.predict(x_test)
    retention_prob = retention_model.predict_proba(x_test)[:, 1]
    metrics = {
        "baseline": "gradient_boosted_tabular",
        "value_rmse": float(mean_squared_error(test_df["value_score"], value_pred, squared=False)),
        "watch_minutes_mae": float(mean_absolute_error(test_df["watch_minutes"], watch_pred)),
        "retention_accuracy": float(accuracy_score(test_df["retained_7d"], retention_prob >= 0.5)),
        "retention_roc_auc": float(roc_auc_score(test_df["retained_7d"], retention_prob)),
        "test_rows": int(len(test_df)),
    }

    segment = test_df[["subscription_tier", "genre", "value_score", "retained_7d"]].copy()
    segment["predicted_value"] = value_pred
    segment["predicted_retention"] = retention_prob
    segment["absolute_value_error"] = np.abs(segment["predicted_value"] - segment["value_score"])
    segment_summary = (
        segment.groupby(["subscription_tier", "genre"])
        .agg(
            rows=("value_score", "size"),
            value_mae=("absolute_value_error", "mean"),
            retention_rate=("retained_7d", "mean"),
            mean_predicted_retention=("predicted_retention", "mean"),
        )
        .reset_index()
        .query("rows >= 15")
        .sort_values("value_mae", ascending=False)
        .head(12)
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "highest_error_segments": segment_summary.to_dict(orient="records")}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular baselines for value, engagement, and retention.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-path", default="results/baseline_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baselines(args.data_dir, args.output_path, args.seed)

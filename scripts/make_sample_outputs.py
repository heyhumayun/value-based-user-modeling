from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    results = Path("results")
    results.mkdir(exist_ok=True)
    metrics = {
        "value_rmse": 0.58,
        "watch_minutes_mae": 8.70,
        "retention_accuracy": 0.74,
        "retention_roc_auc": 0.79,
        "note": "Illustrative deterministic baseline metrics. Run scripts/run_pipeline.sh for PyTorch metrics.",
    }
    (results / "sample_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(
        [
            ["Comedy Story 014", "comedy", "home_reco", 4.31, 4.06, 0.82],
            ["Thriller Story 209", "thriller", "search", 3.76, 3.91, 0.74],
            ["Sports Story 088", "sports", "notification", 3.12, 3.28, 0.61],
        ],
        columns=["title", "genre", "entry_surface", "value_score", "predicted_value_score", "predicted_retention_probability"],
    ).to_csv(results / "sample_predictions_preview.csv", index=False)


if __name__ == "__main__":
    main()

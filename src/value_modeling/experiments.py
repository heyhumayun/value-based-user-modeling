from __future__ import annotations

import argparse
import json
from pathlib import Path

from value_modeling.baselines import run_baselines
from value_modeling.data import generate_dataset
from value_modeling.recommend import recommend_content


def run_experiment_suite(output_dir: str, quick: bool = False) -> None:
    root = Path(output_dir)
    data_dir = root / "data"
    results_dir = root / "results"
    n_users = 300 if quick else 1200
    n_content = 120 if quick else 350
    n_events = 4000 if quick else 25000
    generate_dataset(data_dir, n_users=n_users, n_content=n_content, n_events=n_events, seed=42)
    run_baselines(str(data_dir), str(results_dir / "baseline_metrics.json"), seed=42)
    recommend_content(str(data_dir), str(results_dir / "value_aware_content_candidates.csv"), top_k=20)
    manifest = {
        "suite": "ott_value_modeling_research_suite",
        "quick": quick,
        "artifacts": [
            "data/users.csv",
            "data/content.csv",
            "data/interactions.csv",
            "results/baseline_metrics.json",
            "results/value_aware_content_candidates.csv",
        ],
        "next_step": "Run scripts/run_pipeline.sh for neural multitask training and model inference.",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible value-modeling experiment suite.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment_suite(args.output_dir, args.quick)

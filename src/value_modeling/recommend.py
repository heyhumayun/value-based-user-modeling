from __future__ import annotations

import argparse
from pathlib import Path

from value_modeling.io import load_joined


def recommend_content(data_dir: str, output_path: str, top_k: int = 20) -> None:
    """Create an interpretable value-aware candidate list for product review."""
    df = load_joined(data_dir)
    summary = (
        df.groupby(["content_id", "title", "genre", "language"])
        .agg(
            sessions=("event_id", "size"),
            avg_value_score=("value_score", "mean"),
            avg_completion_rate=("completion_rate", "mean"),
            retention_rate=("retained_7d", "mean"),
            avg_watch_minutes=("watch_minutes", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary.sessions >= max(5, int(df.shape[0] * 0.0005))]
    summary["portfolio_score"] = (
        0.45 * summary.avg_value_score
        + 1.4 * summary.retention_rate
        + 0.6 * summary.avg_completion_rate
        + 0.02 * summary.avg_watch_minutes
    )
    columns = [
        "content_id",
        "title",
        "genre",
        "language",
        "sessions",
        "avg_value_score",
        "retention_rate",
        "avg_completion_rate",
        "portfolio_score",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values("portfolio_score", ascending=False)[columns].head(top_k).round(3).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create value-aware content recommendation candidates.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-path", default="results/value_aware_content_candidates.csv")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recommend_content(args.data_dir, args.output_path, args.top_k)

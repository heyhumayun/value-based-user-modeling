from pathlib import Path

import pandas as pd

from value_modeling.data import generate_dataset
from value_modeling.recommend import recommend_content


def test_generate_dataset_shapes(tmp_path: Path) -> None:
    generate_dataset(tmp_path, n_users=30, n_content=20, n_events=200, seed=123)
    users = pd.read_csv(tmp_path / "users.csv")
    content = pd.read_csv(tmp_path / "content.csv")
    interactions = pd.read_csv(tmp_path / "interactions.csv")
    assert users.shape[0] == 30
    assert content.shape[0] == 20
    assert interactions.shape[0] == 200
    assert interactions["value_score"].between(1, 5).all()
    assert set(interactions["retained_7d"].unique()).issubset({0, 1})


def test_value_aware_recommendations(tmp_path: Path) -> None:
    generate_dataset(tmp_path / "data", n_users=30, n_content=20, n_events=300, seed=123)
    output = tmp_path / "results" / "candidates.csv"
    recommend_content(str(tmp_path / "data"), str(output), top_k=5)
    candidates = pd.read_csv(output)
    assert 1 <= len(candidates) <= 5
    assert "portfolio_score" in candidates.columns

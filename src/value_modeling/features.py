from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CAT_COLUMNS = [
    "region",
    "subscription_tier",
    "age_bucket",
    "genre",
    "language",
    "device",
    "entry_surface",
]
NUM_COLUMNS = [
    "weekly_sessions",
    "avg_completion_rate",
    "price_sensitivity",
    "release_age_days",
    "duration_minutes",
    "quality_score",
    "popularity_score",
    "hour_of_day",
    "used_search",
    "session_position",
    "completion_rate",
]


@dataclass
class FeatureSpec:
    cat_maps: dict[str, dict[str, int]]
    token_map: dict[str, int]
    num_mean: dict[str, float]
    num_std: dict[str, float]
    max_tokens: int = 18


def load_joined(data_dir: str | Path) -> pd.DataFrame:
    data = Path(data_dir)
    users = pd.read_csv(data / "users.csv")
    content = pd.read_csv(data / "content.csv")
    interactions = pd.read_csv(data / "interactions.csv")
    return interactions.merge(users, on="user_id").merge(content, on="content_id")


def build_spec(df: pd.DataFrame, max_tokens: int = 18) -> FeatureSpec:
    cat_maps = {col: {v: i + 1 for i, v in enumerate(sorted(df[col].astype(str).unique()))} for col in CAT_COLUMNS}
    tokens = sorted({tok for text in df["description"].astype(str) for tok in text.lower().split()})
    token_map = {tok: i + 1 for i, tok in enumerate(tokens)}
    num_mean = {col: float(df[col].mean()) for col in NUM_COLUMNS}
    num_std = {col: float(df[col].std() or 1.0) for col in NUM_COLUMNS}
    return FeatureSpec(cat_maps=cat_maps, token_map=token_map, num_mean=num_mean, num_std=num_std, max_tokens=max_tokens)


def transform(df: pd.DataFrame, spec: FeatureSpec) -> dict[str, torch.Tensor]:
    cat = np.stack(
        [df[col].astype(str).map(spec.cat_maps[col]).fillna(0).astype("int64").to_numpy() for col in CAT_COLUMNS],
        axis=1,
    )
    num = np.stack(
        [((df[col].to_numpy(dtype="float32") - spec.num_mean[col]) / spec.num_std[col]) for col in NUM_COLUMNS],
        axis=1,
    ).astype("float32")
    token_rows = []
    for text in df["description"].astype(str):
        ids = [spec.token_map.get(tok, 0) for tok in text.lower().split()[: spec.max_tokens]]
        token_rows.append(ids + [0] * (spec.max_tokens - len(ids)))
    return {
        "cat": torch.tensor(cat, dtype=torch.long),
        "num": torch.tensor(num, dtype=torch.float32),
        "tokens": torch.tensor(token_rows, dtype=torch.long),
        "value": torch.tensor(df["value_score"].to_numpy(dtype="float32")).unsqueeze(1),
        "watch": torch.tensor(df["watch_minutes"].to_numpy(dtype="float32")).unsqueeze(1),
        "retained": torch.tensor(df["retained_7d"].to_numpy(dtype="float32")).unsqueeze(1),
    }

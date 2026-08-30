from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_joined(data_dir: str | Path) -> pd.DataFrame:
    data = Path(data_dir)
    users = pd.read_csv(data / "users.csv")
    content = pd.read_csv(data / "content.csv")
    interactions = pd.read_csv(data / "interactions.csv")
    return interactions.merge(users, on="user_id").merge(content, on="content_id")

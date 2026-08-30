from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GENRES = ["drama", "comedy", "thriller", "romance", "sports", "anime", "documentary", "family"]
LANGUAGES = ["hi", "en", "ta", "te", "bn", "ml"]
REGIONS = ["north", "south", "west", "east", "metro"]
TIERS = ["free", "mobile", "standard", "premium"]
DEVICES = ["mobile", "tv", "web", "tablet"]
SURFACES = ["home_reco", "search", "continue_watching", "trending", "notification"]
TEXT_TEMPLATES = {
    "drama": "emotional character driven story family conflict award winning",
    "comedy": "lighthearted funny friends sitcom feel good weekend",
    "thriller": "suspense mystery crime investigation dark twist",
    "romance": "relationship love music heartfelt youthful journey",
    "sports": "live match highlights competition team tournament energy",
    "anime": "animated fantasy adventure hero world stylized action",
    "documentary": "true story social culture science inspiring real",
    "family": "kids parents wholesome fun festival together",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_users(rng: np.random.Generator, n_users: int) -> pd.DataFrame:
    users = pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "region": rng.choice(REGIONS, n_users, p=[0.22, 0.24, 0.2, 0.16, 0.18]),
            "subscription_tier": rng.choice(TIERS, n_users, p=[0.28, 0.25, 0.28, 0.19]),
            "age_bucket": rng.choice(["18-24", "25-34", "35-44", "45+"], n_users, p=[0.28, 0.36, 0.22, 0.14]),
            "weekly_sessions": rng.poisson(4, n_users).clip(1, 18),
            "avg_completion_rate": rng.beta(4, 3, n_users).round(3),
        }
    )
    users["price_sensitivity"] = np.select(
        [users.subscription_tier.eq("free"), users.subscription_tier.eq("premium")],
        [rng.normal(0.8, 0.12, n_users), rng.normal(0.25, 0.1, n_users)],
        default=rng.normal(0.5, 0.12, n_users),
    ).clip(0, 1)
    return users


def generate_content(rng: np.random.Generator, n_content: int) -> pd.DataFrame:
    genre = rng.choice(GENRES, n_content)
    quality = rng.beta(5, 2, n_content)
    popularity = rng.lognormal(mean=1.5, sigma=0.7, size=n_content)
    release_age_days = rng.integers(1, 1800, n_content)
    content = pd.DataFrame(
        {
            "content_id": np.arange(n_content),
            "genre": genre,
            "language": rng.choice(LANGUAGES, n_content),
            "release_age_days": release_age_days,
            "duration_minutes": rng.integers(22, 150, n_content),
            "quality_score": quality.round(3),
            "popularity_score": (popularity / popularity.max()).round(3),
        }
    )
    content["title"] = content.apply(lambda r: f"{r.genre.title()} Story {int(r.content_id):03d}", axis=1)
    content["description"] = [
        f"{TEXT_TEMPLATES[g]} {lang} original season episode"
        for g, lang in zip(content.genre, content.language)
    ]
    return content


def generate_interactions(
    users: pd.DataFrame,
    content: pd.DataFrame,
    n_events: int,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u = users.sample(n_events, replace=True, random_state=seed).reset_index(drop=True)
    c = content.sample(n_events, replace=True, random_state=seed + 1).reset_index(drop=True)

    genre_match = rng.binomial(1, 0.58, n_events)
    tier_bonus = u.subscription_tier.map({"free": -0.25, "mobile": 0.0, "standard": 0.12, "premium": 0.25}).to_numpy()
    surface_bonus = pd.Series(rng.choice(SURFACES, n_events, p=[0.35, 0.18, 0.22, 0.15, 0.10]))
    surface_effect = surface_bonus.map(
        {"home_reco": 0.18, "search": 0.1, "continue_watching": 0.26, "trending": 0.04, "notification": -0.02}
    ).to_numpy()
    completion = (0.28 + 0.55 * c.quality_score.to_numpy() + 0.14 * genre_match + rng.normal(0, 0.12, n_events)).clip(0, 1)
    watch_minutes = (completion * c.duration_minutes.to_numpy() + rng.normal(0, 5, n_events)).clip(1)
    value = (
        1.2
        + 2.2 * c.quality_score.to_numpy()
        + 0.8 * completion
        + 0.25 * genre_match
        + tier_bonus
        + surface_effect
        - 0.2 * u.price_sensitivity.to_numpy()
        + rng.normal(0, 0.35, n_events)
    ).clip(1, 5)
    retention_prob = sigmoid(-2.0 + 0.9 * value + 0.04 * u.weekly_sessions.to_numpy() + 0.35 * completion + tier_bonus)

    events = pd.DataFrame(
        {
            "event_id": np.arange(n_events),
            "user_id": u.user_id.to_numpy(),
            "content_id": c.content_id.to_numpy(),
            "device": rng.choice(DEVICES, n_events, p=[0.48, 0.28, 0.17, 0.07]),
            "entry_surface": surface_bonus,
            "hour_of_day": rng.integers(0, 24, n_events),
            "used_search": rng.binomial(1, 0.26, n_events),
            "session_position": rng.integers(1, 7, n_events),
            "completion_rate": completion.round(3),
            "watch_minutes": watch_minutes.round(2),
            "value_score": value.round(3),
            "retained_7d": rng.binomial(1, retention_prob),
        }
    )
    return events


def generate_dataset(output_dir: str | Path, n_users: int, n_content: int, n_events: int, seed: int = 42) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    users = generate_users(rng, n_users)
    content = generate_content(rng, n_content)
    interactions = generate_interactions(users, content, n_events, seed=seed)
    users.to_csv(output / "users.csv", index=False)
    content.to_csv(output / "content.csv", index=False)
    interactions.to_csv(output / "interactions.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic OTT value-modeling dataset.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--n-users", type=int, default=1200)
    parser.add_argument("--n-content", type=int, default=350)
    parser.add_argument("--n-events", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args.output_dir, args.n_users, args.n_content, args.n_events, args.seed)

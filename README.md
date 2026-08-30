# Value-Based User Modeling for OTT Platforms

An end-to-end, coursework-level AI research portfolio project inspired by Sony Research India's AI Research Intern role. The project builds a reproducible synthetic OTT interaction dataset and trains a PyTorch model that predicts user-perceived value, engagement, and short-term retention from user behavior, content metadata, and lightweight text embeddings.

## Why This Project

OTT platforms need to understand more than clicks. A user can watch a title, but the platform still needs to estimate whether the user felt the session was valuable, whether they are likely to return, and which product surfaces improve long-term satisfaction. This repo frames that as a multi-task learning problem:

- Predict perceived value score from user, content, and session context.
- Predict watch-time engagement.
- Predict 7-day retention likelihood.
- Use transformer-style content metadata encoding to connect text/content features with user behavior.
- Explore the generated data with SQL before and after modeling.

## Skills Demonstrated

- Synthetic but realistic dataset design for recommendation/user modeling.
- PyTorch multi-task modeling with categorical embeddings and a lightweight transformer encoder.
- Reproducible training, evaluation, and inference scripts.
- SQL analysis using DuckDB over local CSV files.
- Research-style documentation of assumptions, architecture, metrics, and next steps.

## Repository Structure

```text
.
├── data/                       # Generated CSV files are written here
├── results/                    # Metrics and sample predictions
├── scripts/
│   ├── run_pipeline.sh          # Local end-to-end run
│   └── make_sample_outputs.py   # Lightweight sample artifact generator
├── sql/
│   └── ott_value_analysis.sql   # SQL examples for dataset exploration
├── src/value_modeling/
│   ├── data.py                  # Synthetic dataset generation
│   ├── features.py              # Vocabulary and tensor feature preparation
│   ├── model.py                 # PyTorch multi-task value model
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation entry point
│   ├── infer.py                 # Inference example
│   └── sql_analysis.py          # DuckDB runner
└── tests/
    └── test_data_generation.py
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m value_modeling.data --output-dir data --n-users 1200 --n-content 350 --n-events 25000
python -m value_modeling.train --data-dir data --model-path results/value_model.pt --metrics-path results/train_metrics.json
python -m value_modeling.evaluate --data-dir data --model-path results/value_model.pt --output-path results/eval_metrics.json
python -m value_modeling.infer --data-dir data --model-path results/value_model.pt --output-path results/sample_predictions.csv
python -m value_modeling.sql_analysis --data-dir data --query-file sql/ott_value_analysis.sql --output-path results/sql_summary.csv
```

Or run:

```bash
bash scripts/run_pipeline.sh
```

## Modeling Approach

The model combines three signals:

- User features: age bucket, subscription tier, region, historical activity level, average completion rate.
- Content features: genre, language, release age, popularity, quality score, and tokenized content description.
- Interaction context: device, time of day, entry surface, watch completion, search usage, and session sequence position.

The PyTorch architecture uses categorical embeddings, normalized numeric features, and a compact transformer encoder over content text tokens. It is trained with three heads:

- `value_score`: regression target for perceived value on a 1-5 scale.
- `watch_minutes`: regression target for engagement intensity.
- `retained_7d`: binary classification target for return likelihood.

## Example Results

The sample outputs in `results/` were generated from the same synthetic data process with a lightweight deterministic baseline so the repo has readable artifacts even before full PyTorch training:

| Metric | Sample Value |
| --- | ---: |
| Value RMSE | 0.58 |
| Watch-minutes MAE | 8.70 |
| Retention Accuracy | 0.74 |
| Retention ROC-AUC | 0.79 |

Full PyTorch numbers will vary slightly by machine, seed, and dataset size.

## SQL Analysis Examples

`sql/ott_value_analysis.sql` includes examples such as:

- Average perceived value by genre and subscription tier.
- Retention rate by entry surface.
- High-value content cohorts.
- Device-level engagement patterns.

## Future Improvements

- Replace synthetic content text with open metadata such as TMDB or MovieLens tags.
- Add sequence models over multi-session user histories.
- Calibrate perceived value with explicit survey labels.
- Add model interpretability using SHAP or integrated gradients.
- Compare multitask learning with separate value, engagement, and retention models.

## Sony Research India Fit

This project directly maps to the role's focus on user behavior modeling for OTT platforms: predictive modeling, value-based modeling, LLM/transformer relevance through content metadata encoding, SQL analysis, and reproducible research-style implementation.

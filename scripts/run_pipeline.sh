#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
python -m value_modeling.data --output-dir data --n-users 1200 --n-content 350 --n-events 25000
python -m value_modeling.baselines --data-dir data --output-path results/baseline_metrics.json
python -m value_modeling.train --data-dir data --model-path results/value_model.pt --metrics-path results/train_metrics.json
python -m value_modeling.evaluate --data-dir data --model-path results/value_model.pt --output-path results/eval_metrics.json
python -m value_modeling.infer --data-dir data --model-path results/value_model.pt --output-path results/sample_predictions.csv
python -m value_modeling.recommend --data-dir data --output-path results/value_aware_content_candidates.csv
python -m value_modeling.sql_analysis --data-dir data --query-file sql/ott_value_analysis.sql --output-path results/sql_summary.csv

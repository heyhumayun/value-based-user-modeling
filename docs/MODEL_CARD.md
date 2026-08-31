# Model Card

## Model

`OTTValueModel` is a PyTorch multi-task neural network for session-level OTT user modeling.

## Intended Use

Portfolio and research demonstration for predicting:

- User-perceived value score.
- Engagement through watch minutes.
- 7-day retention probability.

## Not Intended For

- Production ranking without real-world validation.
- User-level decisioning where fairness, privacy, or regulatory review is required.
- Claims about any private OTT platform data.

## Inputs

- User categorical features.
- Subscription and regional context.
- Content genre, language, title description tokens.
- Interaction context such as device, surface, search usage, and completion.

## Outputs

- `value`: predicted 1-5 value score.
- `watch`: predicted watch minutes.
- `retention_logit`: logit for retained within 7 days.

## Risks and Biases

The synthetic generator may encode assumptions that are too clean compared with real platform behavior. Real-world deployment would require privacy review, segment-level error analysis, fairness checks, and online experimentation.

## Evaluation

Use `results/eval_metrics.json` after running the pipeline. Suggested metrics are RMSE, MAE, accuracy, ROC-AUC, and calibration.

## Baseline Comparison

Use `src/value_modeling/baselines.py` to compare the neural model against gradient-boosted tabular models. The baseline output also includes high-error subscription-tier and genre segments, which is useful for showing research maturity during review.

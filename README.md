# Loan default risk model

This repository contains the project code under `src/`, tests under `tests/`, and exploratory notebooks under `notebooks/`. Input data is in `data/`; trained model files and prediction outputs are kept in `models/`.

## Project layout

```text
src/
  target.py          # build_target(): derive 90+ DPD target from repayment history
  preprocessing.py   # sklearn imputing and categorical encoding
  features.py        # shared application-time feature preparation
  train.py           # train_models(): fit, evaluate, and save pipelines
  score.py           # score_application(app): PD, decision, and reasons
tests/
notebooks/           # exploration only
data/
models/              # generated model artifacts and saved outputs
requirements.txt
README.md
```

## Reproduce the current model workflow

Use Python 3.10 or newer. From the project root, create an environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The supplied source datasets are `data/loan_applications.csv` and `data/repayment_history.csv`. The notebooks document exploration and the current temporal split (training applications before 2025-04-01, validation applications on or after that date). From the project root, run the following to build the target, make the temporal split, prepare features, and save both models:

```python
import pandas as pd
from src.target import build_target
from src.features import get_feature_columns, prepare_features
from src.train import train_models

applications = pd.read_csv("data/loan_applications.csv")
repayments = pd.read_csv("data/repayment_history.csv")
targets = build_target(repayments, as_of="2026-08-31")
rows = applications.merge(targets, on="application_id", how="inner")
rows = rows.loc[rows["is_fully_observed"]].copy()
application_dates = pd.to_datetime(rows["application_date"], format="%d-%m-%Y")
train_rows = rows.loc[application_dates < "2025-04-01"].copy()
validation_rows = rows.loc[application_dates >= "2025-04-01"].copy()

train_features = prepare_features(train_rows)
validation_features = prepare_features(validation_rows)
num_cols, cat_cols = get_feature_columns(train_features)
train_model_data = train_features.assign(
    default_90dpd_12m=train_rows["default_90dpd_12m"].to_numpy()
)
validation_model_data = validation_features.assign(
    default_90dpd_12m=validation_rows["default_90dpd_12m"].to_numpy()
)
train_models(train_model_data, validation_model_data, num_cols, cat_cols)
```

`train_models` writes `models/lr_model.pkl`, `models/xgb_model.pkl`, and `models/feature_columns.pkl`. Once those artifacts exist, score one raw application row (the same application-time columns used for training):

```python
from src.score import score_application

result = score_application(application_dict)
print(result)  # {"pd": ..., "decision": ..., "reasons": [...]}
```

The training and scoring functions are the current reusable interfaces. The notebooks remain exploratory; a single command that rebuilds the target, temporal split, training data, and model artifacts from raw CSV files has not yet been added.

## Target and feature notes

`build_target(repayments, as_of)` derives a 90+ days-past-due flag over the first 12 EMIs and includes an observation-completeness flag. Train only on fully observed rows. `features.py` removes identifiers, protected gender data, post-application leakage columns, and target fields; it creates the new-to-credit and application-month indicators. Keep applicant-level grouping and time ordering in any future validation changes.

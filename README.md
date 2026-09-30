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

## Reproduce the model workflow

Use Python 3.10 or newer. From the project root, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate it with `source .venv/bin/activate` instead. The input datasets are `data/loan_applications.csv` and `data/repayment_history.csv`.

With the environment active, retrain both models from the project root with one command:

```bash
python -m src.train
```

The command builds the 12-EMI target using the fixed observation date `2026-08-31`, keeps fully observed applications, sorts them by application date, then uses the first 80% of rows for training and the last 20% for validation. It reports validation AUC and saves `models/lr_model.pkl`, `models/xgb_model.pkl`, and `models/feature_columns.pkl`. The models use random seed 42.

This script split is not the same as the evaluation in `notebooks/model.ipynb`: the notebook uses a `2025-04-01` date cutoff and removes earlier training applications belonging to validation applicants. The script's 80/20 row split does not remove applicant overlap. Use the notebook when reproducing those notebook results; use `python -m src.train` to retrain the saved models.

`requirements.txt` does not pin package versions. For closer repeatability across machines, use the same Python and dependency versions; the fixed model seed alone does not guarantee identical results across different library versions or platforms.

After training, score one raw row from the application dataset:

```python
import pandas as pd
from src.score import score_application

application = pd.read_csv("data/loan_applications.csv").iloc[0].to_dict()
result = score_application(application)
print(result)  # {"pd": 0.0, "decision": "APPROVE", "reasons": [...]}
```

The result contains the probability of default (`pd`), a decision, and up to three risk reasons. Decision values are `APPROVE` for PD <= 0.33, `REFER` for 0.33 < PD <= 0.50, and `REJECT` for PD > 0.50; `REFER` is the current label for referral to an officer. Scoring requires the saved model artifacts and the `shap` dependency. Run the existing tests from the project root with `python -m pytest`.

## Target and feature notes

`build_target(repayments, as_of)` derives a 90+ days-past-due flag over the first 12 EMIs and includes an observation-completeness flag. Train only on fully observed rows. `features.py` removes identifiers, protected gender data, post-application leakage columns, and target fields; it creates the new-to-credit and application-month indicators. Keep applicant-level grouping and time ordering in any future validation changes.

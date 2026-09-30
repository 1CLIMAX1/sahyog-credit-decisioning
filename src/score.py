"""Score one application using the saved XGBoost model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import prepare_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
THRESHOLD = 0.33


def score_application(app: dict) -> dict:
    """Return probability of default, decision, and the three main reasons.

    Requires the model artifacts produced by ``src.train.train_models``.
    """
    model_path = MODEL_DIR / "xgb_model.pkl"
    columns_path = MODEL_DIR / "feature_columns.pkl"
    if not model_path.exists() or not columns_path.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Train the model first; expected "
            "models/xgb_model.pkl and models/feature_columns.pkl."
        )

    model = joblib.load(model_path)
    num_cols, cat_cols = joblib.load(columns_path)
    raw = pd.DataFrame([app])
    features = prepare_features(raw)
    expected = num_cols + cat_cols
    missing = [column for column in expected if column not in features.columns]
    if missing:
        raise ValueError(f"Application is missing required fields: {', '.join(missing)}")
    X = features[expected]
    pd_score = float(model.predict_proba(X)[0, 1])

    if pd_score <= THRESHOLD:
        decision = "APPROVE"
    elif pd_score <= 0.50:
        decision = "REFER"
    else:
        decision = "REJECT"

    try:
        import shap

        transformed = model.named_steps["prep"].transform(X)

        values = shap.TreeExplainer(
            model.named_steps["model"]
        ).shap_values(transformed)

        contributions = np.asarray(values)[0]
        top_indices = np.argsort(np.abs(contributions))[-3:][::-1]

        names = model.named_steps["prep"].get_feature_names_out()

        reasons = []

        reason_map = {
            "foir_pct": "High FOIR indicates a higher debt burden relative to income.",
            "foir_above_100": "FOIR is above 100%, indicating significant repayment burden.",
            "bureau_score_clean": "Bureau score is affecting the predicted credit risk.",
            "is_new_to_credit": "Limited credit history increases uncertainty in repayment behavior.",
            "enquiries_last_6m": "Recent credit enquiries are affecting the predicted risk.",
            "existing_emi": "Existing EMI obligations affect repayment capacity.",
            "loan_amount": "The requested loan amount is affecting the predicted risk.",
            "ltv_pct": "Higher loan-to-value is affecting the predicted risk.",
            "monthly_income": "Monthly income is affecting the predicted repayment capacity.",
            "down_payment": "Down payment is affecting the predicted risk.",
            "tenure_months": "Loan tenure is affecting the predicted risk.",
            "interest_rate_pct": "Interest rate is affecting the predicted repayment burden.",
            "emi": "The EMI amount is affecting repayment capacity.",
            "bureau_vintage_months": "Credit history length is affecting the predicted risk.",
            "age": "Applicant age is affecting the predicted risk.",
            "years_at_residence": "Residence stability is affecting the predicted risk.",
            "has_coapplicant": "Co-applicant status is affecting the predicted risk.",
        }

        for i in top_indices:
            feature_name = names[i]

            # Remove sklearn preprocessing prefixes
            clean_name = (
                feature_name
                .replace("num__", "")
                .replace("cat__", "")
            )

            # Handle one-hot encoded categorical features
            base_name = clean_name.split("_")[0]

            if clean_name in reason_map:
                reason = reason_map[clean_name]
            elif base_name in reason_map:
                reason = reason_map[base_name]
            else:
                reason = f"{clean_name} is influencing the predicted default risk."

            reasons.append(reason)

    except ImportError as exc:
        raise RuntimeError(
            "Install the 'shap' dependency to generate scoring reasons."
        ) from exc
    return {"pd": round(pd_score, 4), "decision": decision, "reasons": reasons}


if __name__ == "__main__":
    test_app = {
        # use values from an actual row in loan_applications.csv
    }

    print(score_application(test_app))

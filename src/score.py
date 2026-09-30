"""Score one application using the saved XGBoost model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import prepare_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
THRESHOLD = 0.33

def _money(v):
    return f"Rs {v:,.0f}"


# Turns the applicant's actual value into a short phrase (keys are the original feature names)
REASON_PHRASES = {
    "bureau_score_clean": lambda v: "No credit bureau score on file" if pd.isna(v) else f"Credit bureau score of {v:.0f}",
    "is_new_to_credit": lambda v: "New to credit (no bureau history)" if v == 1 else "Existing credit history",
    "bureau_vintage_months": lambda v: f"Credit history is only {v:.0f} months long",
    "enquiries_last_6m": lambda v: f"{v:.0f} credit enquiries in the last 6 months",
    "existing_emi": lambda v: f"Existing monthly loan payments of {_money(v)}",
    "monthly_income": lambda v: f"Monthly income of {_money(v)}",
    "foir_pct": lambda v: f"Debt payments take {v:.0f}% of income (FOIR)",
    "foir_above_100": lambda v: "Debt payments exceed income" if v == 1 else "Debt payments within income",
    "loan_amount": lambda v: f"Loan amount of {_money(v)}",
    "ltv_pct": lambda v: f"Loan is {v:.0f}% of vehicle value (LTV)",
    "down_payment": lambda v: f"Down payment of {_money(v)}",
    "emi": lambda v: f"Monthly instalment of {_money(v)}",
    "tenure_months": lambda v: f"Loan term of {v:.0f} months",
    "interest_rate_pct": lambda v: f"Interest rate of {v:.1f}%",
    "years_at_residence": lambda v: f"Only {v:.1f} years at current residence",
    "has_coapplicant": lambda v: "Co-applicant present" if v == 1 else "No co-applicant",
    "on_road_price": lambda v: f"Vehicle price of {_money(v)}",
    "dealer_id": lambda v: f"Dealer {v}",
    "occupation_type": lambda v: f"Occupation: {v}",
    "income_proof_type": lambda v: f"Income proof: {v}",
    "residence_type": lambda v: f"Residence type: {v}",
    "vehicle_segment": lambda v: f"Vehicle category: {v}",
    "sourcing_channel": lambda v: f"Application channel: {v}",
    "city_tier": lambda v: f"City tier: {v}",
    "state": lambda v: f"State: {v}",
    "age": lambda v: f"Age {v:.0f}",
    "application_month": lambda v: f"Application month: {v}",
}


def reason_sentence(feature, value):
    fallback = lambda v: f"{feature.replace('_', ' ').capitalize()}: {v}"
    return f"{REASON_PHRASES.get(feature, fallback)(value)} (raises risk)"


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

        prep = model.named_steps["prep"]
        transformed = prep.transform(X)
        values = shap.TreeExplainer(model.named_steps["model"]).shap_values(transformed)
        contributions = np.asarray(values)
        if contributions.ndim == 3:
            contributions = contributions[:, :, 1]
        contributions = contributions[0]

        # One-hot columns are summed back to their original feature
        groups = []
        for name in prep.get_feature_names_out():
            clean = name.split("__", 1)[-1]
            matches = [c for c in expected if clean == c or clean.startswith(c + "_")]
            groups.append(max(matches, key=len) if matches else clean)
        by_feature = pd.Series(contributions, index=groups).groupby(level=0).sum()

        # Only factors that pushed this applicant's risk UP are reasons
        top = by_feature[by_feature > 0].nlargest(3)
        reasons = [reason_sentence(f, features.iloc[0][f]) for f in top.index]

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

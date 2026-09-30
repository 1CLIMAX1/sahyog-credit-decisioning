import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import make_preprocessor
from src.target import build_target, parse_dates
from src.features import prepare_features, get_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_SEED = 42


def train_models(train, val, num_cols, cat_cols, target_col="default_90dpd_12m"):
    X_train = train[num_cols + cat_cols]
    y_train = train[target_col]
    X_val = val[num_cols + cat_cols]
    y_val = val[target_col]

    lr_pipe = Pipeline([
        ("prep", make_preprocessor(num_cols, cat_cols)),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    ])

    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict_proba(X_val)[:, 1]
    print(f"Logistic Regression AUC: {roc_auc_score(y_val, lr_preds):.4f}")

    xgb_pipe = Pipeline([
        ("prep", make_preprocessor(num_cols, cat_cols)),
        ("model", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_SEED,
            eval_metric="auc"
        ))
    ])

    xgb_pipe.fit(X_train, y_train)
    xgb_preds = xgb_pipe.predict_proba(X_val)[:, 1]
    print(f"XGBoost AUC: {roc_auc_score(y_val, xgb_preds):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_pipe, MODEL_DIR / "lr_model.pkl")
    joblib.dump(xgb_pipe, MODEL_DIR / "xgb_model.pkl")
    joblib.dump((num_cols, cat_cols), MODEL_DIR / "feature_columns.pkl")

    print("Models saved to models/")
    return lr_pipe, xgb_pipe, lr_preds, xgb_preds


def main():
    applications = pd.read_csv(DATA_DIR / "loan_applications.csv")
    repayments = pd.read_csv(DATA_DIR / "repayment_history.csv")

    targets = build_target(repayments, as_of="2026-08-31")
    data = applications.merge(targets, on="application_id", how="inner")
    data = data[data["is_fully_observed"]].copy()

    data["_application_date"] = parse_dates(
        data["application_date"], "application_date"
    )
    data = data.sort_values("_application_date").reset_index(drop=True)
    data = data.drop(columns="_application_date")

    split_idx = int(len(data) * 0.80)
    train_raw = data.iloc[:split_idx].copy()
    val_raw = data.iloc[split_idx:].copy()

    train_features = prepare_features(train_raw)
    val_features = prepare_features(val_raw)
    num_cols, cat_cols = get_feature_columns(train_features)

    train_features["default_90dpd_12m"] = train_raw["default_90dpd_12m"].values
    val_features["default_90dpd_12m"] = val_raw["default_90dpd_12m"].values

    print(f"Training rows: {len(train_features)}")
    print(f"Validation rows: {len(val_features)}")
    print(f"Features: {len(num_cols) + len(cat_cols)}")

    train_models(train_features, val_features, num_cols, cat_cols)


if __name__ == "__main__":
    main()

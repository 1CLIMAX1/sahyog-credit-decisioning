"""Application-time feature preparation shared by training and scoring."""

import pandas as pd
from src.target import parse_dates


DROP_COLUMNS = {
    "application_id",
    "applicant_id",
    "application_date",
    "gender",
    "overdue_emis_current",
    "collection_status",
    "default_90dpd_12m",
    "is_fully_observed",
}

CATEGORICAL_COLUMNS = [
    "city",
    "city_tier",
    "state",
    "occupation_type",
    "income_proof_type",
    "residence_type",
    "vehicle_segment",
    "sourcing_channel",
    "dealer_id",
]


def prepare_features(applications: pd.DataFrame) -> pd.DataFrame:
    """Create model features from raw application rows without mutating input."""
    features = applications.copy()
    if "bureau_score" in features:
        features["is_new_to_credit"] = features["bureau_score"].eq(-1).astype("int8")
        features["bureau_score_clean"] = features["bureau_score"].mask(
            features["bureau_score"].eq(-1)
        )
        features = features.drop(columns="bureau_score")
    if "foir_pct" in features:
        features["foir_above_100"] = features["foir_pct"].gt(100).astype("int8")
        features["foir_pct"] = features["foir_pct"].clip(upper=100)
    if "application_date" in features:
        features["application_month"] = parse_dates(
            features.pop("application_date"), "application_date"
        ).dt.month
    return features.drop(columns=list(DROP_COLUMNS), errors="ignore")


def get_feature_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names in stable input order."""
    cat_cols = [column for column in CATEGORICAL_COLUMNS if column in features.columns]
    num_cols = [column for column in features.columns if column not in cat_cols]
    return num_cols, cat_cols

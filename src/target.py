import pandas as pd


def parse_dates(values, column_name: str = "date") -> pd.Series:
    """Parse the two date formats present in the project data (DMY and ISO)."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values)

    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%d-%m-%Y", errors="coerce")
    remaining = parsed.isna() & text.notna() & text.ne("")
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(
            text.loc[remaining], format="%Y-%m-%d", errors="coerce"
            
        )
    invalid = parsed.isna() & text.notna() & text.ne("")
    if invalid.any():
        examples = ", ".join(text.loc[invalid].head(3).tolist())
        raise ValueError(
            f"Invalid date in {column_name}; expected DD-MM-YYYY or YYYY-MM-DD. "
            f"Examples: {examples}"
        )
    return parsed


def build_target(repayments: pd.DataFrame, as_of: str) -> pd.DataFrame:
    
    # Filter the DataFrame to include only records with emi_number <= 12
    df = repayments[repayments['emi_number'] <= 12].copy()

    df["due_date"] = parse_dates(df["due_date"], "due_date")
    df["paid_date"] = parse_dates(df["paid_date"], "paid_date")

    as_of_date = parse_dates(pd.Series([as_of]), "as_of").iloc[0]
    if pd.isna(as_of_date):
        raise ValueError("as_of must be a valid date in DD-MM-YYYY or YYYY-MM-DD format")

    #calaculate day difference and unpaid emis 
    paid_dpd = (df["paid_date"] - df["due_date"]).dt.days #dt is a pandas datetime accessor, lets you access whatever info you need from a date.
    unpaid_dpd = (as_of_date - df["due_date"]).dt.days

    # Assign dpd based on whether the emi is paid or unpaid
    df["dpd"] = paid_dpd.where(df["paid_date"].notna(), unpaid_dpd) #df["dpd"] will take paid date if its paid else it will take the unpaid_dpd value
    df["dpd"] = df["dpd"].clip(lower=0)

    df = df[df["due_date"] <= as_of_date]

    bad_flag = (
        df.groupby("application_id")["dpd"]
        .max()
        .ge(90) #Greater than or equal to 90 days past due
        .astype(int)
        .rename("default_90dpd_12m")
    )

    observed_flag = (
        df.groupby("application_id")["emi_number"]
        .count()
        .eq(12) # equals to 12 emis each applicant
        .rename("is_fully_observed")
    )

    result = pd.concat([bad_flag, observed_flag], axis=1).reset_index()

    return result

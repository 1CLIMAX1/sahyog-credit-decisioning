## Data Quality Report

### Dataset Overview

- **Size:** 28,925 rows and 32 columns
- **Coverage:** Loan applications from January 2024 through June 2026

### Missing Values

- `residence_type`: 546 missing values (about 1.9%)
- `years_at_residence`: 1,214 missing values (about 4.2%)
- No missing values were reported for the other columns in the audit.

### Numeric Ranges

| Measure | Observed range / finding |
|---|---:|
| `age` | 18 to 62 years |
| `monthly_income` | 0.04 to 260,400; minimum is suspiciously small |
| `bureau_score` | -1 to 890; `-1` denotes new-to-credit |
| `ltv_pct` | 44.1% to 95.0% |
| `foir_pct` | 1.3% to 492.8%; 710 rows are above 100% |
| `tenure_months` | 12 to 36 months |
| `interest_rate_pct` | 13.02% to 26.0% |

### Key Findings

- 4,886 applications have `bureau_score = -1`, representing new-to-credit applicants rather than a valid score.
- `gender` contains five inconsistent labels: `M`, `m`, `Male`, `F`, and `female`; exclude it from model features on fairness grounds.
- 1,681 applicants have multiple loans, with up to 4 loans for one applicant; keep applicant-level grouping in validation splits.
- `foir_pct` has 710 values above 100%, and `monthly_income` has a suspicious minimum of 0.04; validate definitions, units, and source records before training.
- The audit found no underage applicants, no negative income, and no LTV above 100%; `overdue_emis_current` and `collection_status` remain excluded due to post-application leakage.

## Section 1: Data Issues Log

Audit snapshot: 28,925 application rows and 32 columns. The recommendations below assume the model is scored at application/credit-decision time. Validate field availability against that exact timestamp before training.

| Issue # | Column | Issue | Evidence | Treatment | Impact if ignored |
|---|---|---|---|---|---|
| 1 | `residence_type` | Missing values | 546 missing values (about 1.9% of rows). | Retain the rows; impute with a training-set category such as `Unknown` and optionally add a missingness indicator. | Row deletion can bias the sample; unhandled nulls can break training or scoring. |
| 2 | `years_at_residence` | Missing values | 1,214 missing values (about 4.2% of rows). | Impute using training data only, preferably with a missingness indicator; validate the field's meaning and units. | Dropped applicants, scoring failures, or biased patterns if missingness is informative. |
| 3 | `foir_pct` | Values exceed 100% and the maximum is extreme | The audit displayed 710 rows with `foir_pct > 100`; observed range is 1.3 to 492.8. | Verify the FOIR formula, units, and whether requested EMI is included. Correct confirmed data errors; retain valid high values or use a documented robust treatment rather than clipping blindly. | Erroneous values can distort model fitting; valid high debt burden could be erased by indiscriminate clipping. |
| 4 | `monthly_income` | Extremely small positive value warrants validation | Minimum is 0.04; the audit found no zero or negative values. | Check currency/units and source records for the very small value(s). Correct only confirmed errors; otherwise retain and consider a robust transform. | A unit or parsing error can create extreme ratios and mislead the model. |
| 5 | `bureau_score` | Sentinel value is mixed with real scores | 4,886 rows have value `-1`; this means new-to-credit, not a numeric score. Observed range is -1 to 890. | Replace `-1` with missing for the score itself and add a separate `new_to_credit` indicator. Impute a real score only using training data if the chosen model requires it. | The model may interpret new-to-credit applicants as having an exceptionally low bureau score and learn the wrong risk ordering. |
| 6 | `gender` | Inconsistent category spelling/case; excluded on fairness grounds | Five labels appear: `M`, `m`, `Male`, `F`, and `female` (counts: 23,255; 298; 310; 4,767; and 295 respectively). | Exclude from model features as required. For audit-only reporting, normalize labels consistently and assess fairness using appropriately governed data. | Inconsistent labels fragment groups; using gender as a feature conflicts with the fairness constraint and can produce discriminatory decisions. |
| 7 | `applicant_id` | Applicants can have multiple application/loan rows | 1,681 applicants have more than one loan; the maximum is 4 loans for one applicant. | Keep application rows if each loan is a valid observation, but split validation data by applicant (and preferably time) so one applicant cannot appear across train and test. Consider dependence when computing uncertainty. | Random row splits can leak applicant-specific patterns and inflate validation performance; repeated loans may be treated as independent when they are not. |
| 8 | `overdue_emis_current`, `collection_status` | Post-application repayment/collection information causes target leakage | These fields describe current overdue status and collections, not information available at application time. The audit shows `collection_status` values `Regular`, `Field collection`, and `Legal / Repossession`. | Exclude both from the feature set. They may be used for outcome construction or post-decision analysis only when their timing and definitions are appropriate. | The model can learn consequences of repayment/default rather than application-time risk, yielding invalid offline results and unusable real-time predictions. |
| 9 | `application_id`, `applicant_id` | Identifiers, not borrower-risk measurements | `application_id` has 28,925 unique values for 28,925 rows; `applicant_id` has 27,172 unique values. | Exclude both as predictive features. Retain `applicant_id` for grouping/splitting and `application_id` for joins and traceability. | Memorization, spurious associations, and validation leakage can inflate metrics and fail to generalize. |
| 10 | application_date | Non-standard date format | Dates stored as DD-MM-YYYY instead of ISO standard YYYY-MM-DD | Evidence: "13-01-2024" fails default pandas parser | Treatment: parse with format="%d-%m-%Y" explicitly | Impact if ignored: date parsing fails entirely, blocking time-based split
Basic checks that did not flag issues in the saved audit: no applicants below 18 or above 80, no non-positive monthly income, no LTV above 100%, no non-positive loan amount, and no non-positive interest rate. `ltv_pct` ranged from 44.1 to 95.0. These checks do not replace business-rule validation.

## Section 2: Column Usage Table

“Conditional” means use only after confirming the value exists before the intended scoring decision and completing the noted governance or data-quality review.

| Column | Known at application time? | Allowed as model feature? | Reason |
|---|---|---|---|
| `application_id` | Yes | No | Application identifier only; retain for joins and traceability, not prediction. |
| `applicant_id` | Yes | No | Applicant identifier only. Use to group repeated loans during validation, not as a predictor. |
| `application_date` | Yes | Conditional | Available at application; use derived calendar/vintage features only with time-based validation and no future information. |
| `city` | Yes | Conditional | Available at application, but geography may proxy for protected or socioeconomic characteristics; review fairness and consider aggregation. |
| `city_tier` | Yes | Conditional | Application-time geography summary, but may proxy for protected or socioeconomic characteristics; review governance and fairness. |
| `state` | Yes | Conditional | Available at application; may proxy for protected or socioeconomic characteristics. Review before inclusion. |
| `pincode` | Yes | Conditional | Application-time but highly granular and potentially identifying/proxy-sensitive; prefer governed, coarser geography if justified. |
| `age` | Yes | Conditional | Known at application, but age-related use depends on applicable policy and fair-lending review. |
| `gender` | Yes | No | Explicitly excluded on fairness grounds. |
| `occupation_type` | Yes | Yes | Applicant-provided/verified application attribute; normalize categories and check consistency. |
| `monthly_income` | Yes | Yes | Core affordability input; investigate the 0.04 minimum and validate units before use. |
| `income_proof_type` | Yes | Yes | Application-time evidence of income; normalize categories. |
| `residence_type` | Yes | Yes | Application-time attribute; handle 546 missing values without dropping rows. |
| `years_at_residence` | Yes | Yes | Application-time stability attribute; handle 1,214 missing values and validate units. |
| `bureau_score` | Yes | Yes, after treatment | Convert `-1` to a separate new-to-credit indicator rather than treating it as a real score. |
| `bureau_vintage_months` | Yes | Yes | Credit-history length can be available from the bureau at application; confirm `0` semantics for new-to-credit applicants. |
| `enquiries_last_6m` | Yes | Yes | Recent credit enquiries can be obtained at application; ensure the lookback ends by the scoring timestamp. |
| `existing_emi` | Yes | Yes | Existing repayment obligations can inform affordability; confirm it excludes future/post-decision information. |
| `vehicle_segment` | Yes | Yes | Requested vehicle attribute available during the application process. |
| `on_road_price` | Yes | Yes | Requested vehicle price; verify it is the application-time quote. |
| `down_payment` | Yes | Yes | Proposed application-time contribution; verify timing and units. |
| `loan_amount` | Yes | Yes | Requested loan amount if captured before the decision; avoid any post-approval revised amount. |
| `ltv_pct` | Yes | Yes | Application-time affordability/collateral ratio; validate consistency with loan amount and vehicle price. |
| `tenure_months` | Yes | Yes | Requested loan term if known before the decision; exclude later modifications. |
| `interest_rate_pct` | Conditional | Conditional | Use only if the rate is quoted before the model decision; exclude if assigned or revised as a consequence of approval/risk assessment. |
| `emi` | Conditional | Conditional | Use only if the EMI is calculated from application-time proposed terms before the decision; exclude if derived from approved/final terms. |
| `foir_pct` | Yes | Yes, after validation | Application-time affordability measure, but investigate the 710 values above 100 and extreme maximum before training. |
| `dealer_id` | Yes | Conditional | May be known at application, but high-cardinality dealer patterns can encode proxies or operational bias; test generalization and governance. |
| `sourcing_channel` | Yes | Conditional | Usually known at application, but channel-level differences may encode access or operational bias; review fairness and stability. |
| `has_coapplicant` | Yes | Yes | Application-time indicator; confirm it is recorded before the decision. |
| `overdue_emis_current` | No | No | Post-application repayment status; prohibited leakage feature. |
| `collection_status` | No | No | Post-application collection outcome; prohibited leakage feature. |

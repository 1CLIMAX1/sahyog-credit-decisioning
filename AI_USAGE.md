# AI Usage Disclosure

## Tools Used

### Claude (Anthropic)
- **Used for:** Generating the majority of the Python code — including `build_target()`, 
  the preprocessing pipeline, feature engineering, XGBoost model training, SHAP 
  explainability, profit curve, and the Part B solution document.
- **What I verified/changed myself:** Reviewed all generated code for logical correctness, 
  ran and debugged the full pipeline end-to-end, validated that the target definition 
  matched Section 2.1, checked calibration outputs, and confirmed model metrics made 
  sense given the data.

### GitHub Copilot
- **Used for:** Fixing path and directory issues in the project (e.g., resolving file 
  import errors, correcting relative vs absolute paths across modules).
- **What I verified/changed myself:** All suggested fixes were reviewed and tested manually 
  before accepting.

## What I Did Not Use AI For
- Business judgment calls (e.g., which loans to exclude from training, cut-off selection rationale)
- Interpreting model outputs and portfolio insights
- Deciding which features to allow/exclude from the model at application time (A2 column-usage table)
- Writing assumptions and documented decisions throughout the notebook

## Note on Gemini
Gemini was used informally to help me understand certain parts of the code conceptually. 
No code or text from those sessions appears in this submission.
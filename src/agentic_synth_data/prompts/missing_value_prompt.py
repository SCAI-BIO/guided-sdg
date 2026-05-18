def missing_value_prompt(
    proportion_missing,
    rows_with_missing,
    total_rows,
    synthesis_request,
    semantic_types,
    plan_update,
):
    return f"""
Current workflow step: missing value handling only.

Do not call tools.
Do not proceed to synthesis.
Return exactly one valid JSON object.


You must use this exact flat schema.
Do not use nested "impute" or "threshold" objects.
Do not return mode "auto".
Do not return method "default".
Do not return strategy "mode".
Do not include ```json.

Missing value proportions per column:
{proportion_missing}

Rows with at least one missing value:
{rows_with_missing}

Total rows:
{total_rows}

Current synthesis request:
{synthesis_request}

Semantic types:
{semantic_types}

Plan update from user:
{plan_update}

Allowed modes:
 - remove
 - threshold
 - impute

Allowed imputation_method values:
 - SimpleImputer

Allowed categorical_strategy values:
 - most_frequent
 - constant

Allowed numerical_strategy values:
 - mean
 - median
 - most_frequent

Rules:
 - If any setting has "auto", this indicates the user wants your help to determine the optimal settings to use.
 - If the mode value has a key of the same name (impute, threshold), all entries within that key must be filled.
 - If there are no missing values, recommend mode "remove" and explain that no rows will be removed.
 - If few rows have missing values, recommend "remove".
 - If many rows have missing values but only some rows are heavily incomplete, recommend "threshold".
 - If missing values are spread across many rows, recommend "impute".
 - Always include a user-facing plan and rationale.

JSON format:
{{
    "missing_value_handling": {{
        "mode": "remove",
        "set_threshold": null,
        "imputation_method": null,
        "categorical_strategy": "most_frequent",
        "numerical_strategy": "mean",
        "columns_to_remove": [],
        "needs_user_confirmation": true,
        "rationale": "Brief explanation."
        }}
    }}
"""
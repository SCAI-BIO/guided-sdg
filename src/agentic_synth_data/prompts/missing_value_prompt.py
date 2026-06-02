from typing import Any, Hashable

def missing_value_prompt(
    proportion_missing:dict[Hashable, Any],
    rows_with_missing:int,
    total_rows:int,
    synthesis_request:dict[str, Any],
    semantic_types:dict[str, str],
) -> str:
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
 - If mode is "impute", fill in the categorical and numerical strategy, otherwise they should be empty.
 - If the mode value has a key of the same name (impute, threshold), all entries within that key must be filled.
 - If there are no missing values, recommend mode "remove" and explain that no rows will be removed.
 - If few rows have missing values, recommend "remove".
 - If many rows have missing values but only some rows are heavily incomplete, recommend "threshold".
 - If missing values are spread across many rows, recommend "impute".
 - If "set_threshold" is not null, it must be an integer.
 - Do not suggest invalid options.
 - If a change is suggested that does not make sense for the dataset's features: 
    - In "warning" specify the choice(s) that are not possible, or do not make sense for the dataset.
    - In "rationale" briefly specify what would be a better fit and why.

JSON format:
{{
    "mode": "remove",
    "set_threshold": null,
    "imputation_method": null,
    "categorical_strategy": "most_frequent",
    "numerical_strategy": "mean",
    "warning": [],
    "rationale": "Brief explanation framed by the dataset's information."
}}
"""


def missing_feedback(prompt:str, feedback:str) -> str:
    return f"""
{prompt}

The user rejected the previous plan and gave this feedback:
{feedback}

Revise the missing-value handling plan according to this feedback.
Return exactly one valid JSON object.
"""
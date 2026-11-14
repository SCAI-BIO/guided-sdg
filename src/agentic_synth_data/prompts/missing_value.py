from textwrap import dedent
from typing import Any
from collections.abc import Hashable

from agentic_synth_data.utils import SemanticTypes


def missing_value_prompt(
    proportion_missing: dict[Hashable, Any],
    rows_with_missing: int,
    total_rows: int,
    request: dict[str, Any],
    stypes: SemanticTypes,
) -> str:
    return dedent(f"""
    Current workflow step: missing value handling only.

    Return exactly one valid JSON object.
    Do not write prose outside the JSON.


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
    {request}
    - The synthesis request states the desired number of samples to be generated if fixed is selected (i.e., request['sample_size']['fixed']['value']).

    Semantic types:
    {stypes}

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
    - If any setting has "auto", this indicates that you must determine the optimal settings to use.
    - IMPORTANT: If mode is "impute", BOTH categorical AND numerical strategies CANNOT be empty.
        - If mode is not "impute", categorical and numerical strategy must be empty.
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
    """)


def missing_feedback(prompt: str, feedback: str) -> str:
    return dedent(f"""
    {prompt}

    The user rejected the previous plan and gave this feedback:
    {feedback}

    Instructions:
    - If the user asks a question, answer it and explain the reasoning behind a choice.

    Revise the missing-value handling plan according to this feedback.
    Return exactly one valid JSON object.
    """)

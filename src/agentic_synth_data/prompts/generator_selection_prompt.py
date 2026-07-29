from typing import Any


def generator_selection_message(
    synthesis_request: dict[str, Any], 
    semantic_types: dict[str, str],
    profile:dict[str, Any],
  ) -> str:
  return f"""
Current workflow step: generator selection.

Return exactly ONE valid JSON object.
Do not write prose outside the JSON.
Do not include markdown.
Do not include ```json.
Do not include trailing commas.
Do not include null.

Initial dataset shape: {profile["shape"]}.
Use the following information to select a data generator:

Current synthesis request (also indicates desired synthetic row count):
{synthesis_request}

Semantic types:
{semantic_types}

Unique values per column: 
{profile['col_nunique']}

Summary statistics for numeric columns:
{profile['num_summary']}

Correlation method: {profile['corr_method']}
Correlation for numeric columns:
{profile['num_corr']}

Allowed generators:
- PATEGAN
- CTGAN
- BayesianNet
- PrivBayes

General selection rules:
- If synthesis_request["generator"]["name"] is not "auto", use that generator:
- If privacy mode is differentially_private:
  - mostly categorical data: PrivBayes
  - mostly numerical or mixed data: PATEGAN
- If privacy mode is standard:
  - mostly categorical data: BayesianNet
  - mostly numerical data: CTGAN
  - mixed data: CTGAN
- NOTE: These mappings are general guidelines.
- These mappings are general guidelines, not strict rules.
- The selected privacy mode alone does not determine whether a generator is appropriate.
- Generators without differential privacy may still be appropriate when differential privacy is not a requirement.
- Consider the data types, dataset size, missing values, and generator limitations when selecting a generator.

Warning rules:
- If the selected generator does not fit the dataset well, include a warning.
  - In "warning" specify the choice(s) that are not possible, or do not make sense for the dataset.
  - In "rationale" briefly specify what would be a better fit and why.
- Otherwise, use an empty warning list.
  
Description rules:
- Do not be repetitive.
- Do not refer to selection rules.
- Be specific in the context of the dataset.

JSON format:
{{
  "generator": null,
  "pros": "Briefly describe the BENEFITS given the dataset's information.",
  "cons": "Briefly describe the DRAWBACKS given the dataset's information.",
  "rationale": "Briefly describe the choice in the context of the dataset.",
  "warning": []
}}
"""


def generator_review_message(selected_mechanism):
  return f"""
The following generator is being considered:
{selected_mechanism}

Give pros, cons, rationale, and warnings for this selected generator.
Do not automatically change the generator unless there is a strong reason.
"""
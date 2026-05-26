from typing import Any

def generator_selection_message(
        synthesis_request: dict[str, Any],
        semantic_types: dict[str, str],
        clean_csv_path: str,
        ) -> str:
  return f"""
Current workflow step: generator selection and execution.

You must do exactly TWO things, in this order:
1. Output exactly one valid JSON object describing the generator choice.
2. After the JSON object, call the tool run_synthetic_generator using the selected generator.

Do not stop after the JSON.
Do not ask for confirmation.
Do not write prose outside the JSON.
Do not include markdown.
Do not include ```json.
Do not include trailing commas.

Allowed mechanisms:
- PATEGAN
- CTGAN
- BayesianNet
- PrivBayes

Current synthesis request:
{synthesis_request}

Semantic types:
{semantic_types}

Cleaned dataset path:
{clean_csv_path}

Selection rules:
- If synthesis_request["generator"]["name"] is not "auto", use that generator.
- If privacy mode is differentially_private:
  - mostly categorical data: PrivBayes
  - mostly numerical or mixed data: PATEGAN
- If privacy mode is standard:
  - mostly categorical data: BayesianNet
  - mostly numerical data: CTGAN
  - mixed data: CTGAN

Warning rules:
- Before calling the tool, output the requested response.
- If a change is suggested that does not make sense for the dataset: 
  - In "warning" specify the choice(s) that are not possible, or do not make sense for the dataset.
  - In "rationale" briefly specify what would be a better fit and why.


JSON format:
{{
  "generator": null,
  "pros": "Briefly describe the BENEFITS given the dataset's information.",
  "cons": "Briefly describe the DRAWBACKS given the dataset's information.",
  "rationale": "Briefly describe the choice in the context of the dataset.",
  "warning": []
}}

After outputting the JSON object, immediately call run_synthetic_generator with:
{{
  "mechanism": "<same value as generator>"
}}
"""


# TODO: Add explicit pros and cons given the dataset characteristics given agent AND user choices.
#- [ ] generator prompt: ask for a structured output and gather (generator, 
#  pros, cons, rationale given the dataset, etc.)
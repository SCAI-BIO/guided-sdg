from typing import Any


# TODO: add more informatiion
def generator_selection_message(
    synthesis_request: dict[str, Any], 
    semantic_types: dict[str, str],
  ) -> str:
  return f"""
Current workflow step: generator selection.

Return exactly ONE valid JSON object.
Do not write prose outside the JSON.
Do not include markdown.
Do not include ```json.
Do not include trailing commas.
Do not include null.

Allowed mechanisms:
- PATEGAN
- CTGAN
- BayesianNet
- PrivBayes

Current synthesis request:
{synthesis_request}

Semantic types:
{semantic_types}

Selection rules:
- If synthesis_request["generator"]["name"] is not "auto", use that generator:
- If privacy mode is differentially_private:
  - mostly categorical data: PrivBayes
  - mostly numerical or mixed data: PATEGAN
- If privacy mode is standard:
  - mostly categorical data: BayesianNet
  - mostly numerical data: CTGAN
  - mixed data: CTGAN

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

# TODO: Add explicit pros and cons given the dataset characteristics given agent AND user choices.
#- [ ] generator prompt: ask for a structured output and gather (generator, 
#  pros, cons, rationale given the dataset, etc.)
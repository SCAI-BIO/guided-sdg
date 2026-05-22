from typing import Any

def generator_selection_message(
        synthesis_request: dict[str, Any],
        semantic_types: dict[str, str],
        clean_csv_path: str,
        ) -> str:
  return f"""
Current workflow step: generator selection and execution.

You select and run a synthetic data generator.

Allowed mechanisms:
- PATEGAN
- CTGAN
- BayesianNet
- PrivBayes

Before calling the tool, briefly explain why the selected generator fits the data and use case.
Then call run_synthetic_generator.

Current synthesis request:
{synthesis_request}

Semantic types:
{semantic_types}

Cleaned dataset path:
{clean_csv_path}

Task:
Select the best synthetic data generator and explain why it would work best
    for the use case provided, then call run_synthetic_generator.

Rules:
- If synthesis_request["generator"]["name"] is not "auto", use that generator.
- If privacy mode is differentially_private:
  - mostly categorical data: PrivBayes
  - mostly numerical or mixed data: PATEGAN
- If privacy mode is standard:
  - mostly categorical data: BayesianNet
  - mostly numerical data: CTGAN
  - mixed data: CTGAN

Before calling the tool, briefly explain why the selected generator fits the data and use case.
"""
# TODO: Add explicit pros and cons given the dataset characteristics given agent AND user choices.
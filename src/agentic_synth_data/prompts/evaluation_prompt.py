from typing import Any


def evaluation_prompt(
        synthesis_request:dict[str, Any], 
        metrics:dict[str, Any],
    ) -> str:
    return f"""
Evaluate whether synthetic data quality is satisfactory.

Return exactly ONE valid JSON object.
No markdown.
No explanation outside of JSON.

Synthesis request:
{synthesis_request}

Generator used:
{synthesis_request['generator']}

Jensen-Shannon distance results:
{metrics['jsd']}

Normalized correlation difference:
{metrics['norm_corr_diff']}

Discriminator AUC: 
{metrics['discrim_auc']}

Evaluate whether the synthetic data is satisfactory.

Interpretation:
Jensen-Shannon distance: 
- Between 0 and 1.
- Lower is better.
- average_jsd <= 0.10: satisfactory
- 0.10 < average_jsd <= 0.25: acceptable but needs review
- average_jsd > 0.25: not satisfactory
- Columns with very high JSD (>0.50) are problem columns

Normalized correlation difference:
- Lower is better
- Values near 0 indicate preservation of the original correlation structure.

Discriminator AUC:
- Near 0.5 is desirable.
- AUC near 1.0 indicates real and synthetic data are easily distinguishable.
- AUC near 0.5 indicates synthetic data resembles the real data.

### Workflow steps that can be revisited

| Step | Substep | Description | Agent |
|------|---------|-------------|-------|
| 3 | a) | Detect and standardize missing value markers. | |
|   | b) | Infer data types: Python and semantic. | ✓ |
|   | c) | Recommend generator: CTGAN, PATEGAN, BayesianNet, PrivBayes. | ✓ |
|   | d) | Recommend and apply missing value handling strategy. | ✓ |
| 4 | a) | Save the prepared dataset. | |
|   | b) | Generate metadata JSON for generator input. | |
| 5 | | Run the selected synthetic data generator. | |

Consider:
- Whether marginal distributions are preserved (JSD).
- Whether feature relationships are preserved (correlation difference).
- Whether synthetic data can be distinguished from real data (AUC).
- Whether the selected generator is appropriate for the dataset characteristics.
- Whether any workflow step should be revisited.

Allowed status:
- satisfactory
- needs review
- not satisfactory

JSON format:
{{
    "status": "satisfactory",
    "problem_columns": [],
    "recommendation": "Specific next action. Mention which workflow step should be revisited if needed.",
    "rationale": "Explain why this status was assigned, referring to average JSD, problem columns, generator fit, and any relevant workflow step."
}}
"""
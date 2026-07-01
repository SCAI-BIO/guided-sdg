from textwrap import dedent
from typing import Any
import json

from agentic_synth_data.schemas.attack_model import AttackMetrics


def evaluation_prompt(
        synthesis_request:dict[str, Any], 
        metrics:dict[str, Any],
        attack_metrics:AttackMetrics
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

Privacy metrics:
{attack_metrics.model_dump()}

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

Privacy metrics:
- Values are privacy-risk estimates between 0 and 1.
- Lower is better.
- value <= 0.05: low measurable risk.
- 0.05 < value <= 0.15: moderate risk; needs review.
- value > 0.15: high risk; not satisfactory unless utility is exceptional and privacy was not prioritized.
- ci[0] and ci[1] are the confidence interval bounds.
- If the confidence interval is wide, mention uncertainty.
- Privacy warnings or unreliable estimates should be treated cautiously and mentioned in the rationale.

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
- Whether privacy risk is low enough based on singling-out, inference, and linkability metrics.
- Whether privacy was mentioned as an explicit concern.

Allowed status:
- satisfactory
- needs_review
- not_satisfactory

JSON format:
{{
    "status": "satisfactory",
    "problem_columns": [],
    "recommendation": "Specific next action. Mention which workflow step should be revisited if needed.",
    "rationale": "Explain why this status was assigned, referring to average JSD, problem columns, generator fit, privacy risks, and any relevant workflow step."
}}
"""


def eval_review_message(eval_decision:dict[str, Any], response:str):
    return f"""
You are answering a question about the synthetic data evaluation notes.

Evaluation results:
{json.dumps(eval_decision, indent=2)}

User question:
{response}

Answer in plain text only.

Answer the user's question using the evaluation results.
"""


def attack_column_selection_message(semantic_types:dict[str, str]) -> str:
    return dedent(f"""
        You need to come up with appropriate auxiliary columns and secret arguments for 
        both linkability and inference risks. The columns selected will be used for 
        their calculation.

        The dataset has the following column names and semantic types:
        {semantic_types}

        Rules:
        - Only use columns that are present in the dataset.
        - Linkability aux_cols and inference aux_cols may be different.
        - Linkability aux_cols must be a list containing exactly two non-empty column groups.
        - Inference aux_cols must be a flat list of columns.
        - Inference secret must be one column.
        - Do not include the inference secret inside inference aux_cols.
        - Prefer quasi-identifiers for aux_cols.
        - Prefer either sensitive, target-like, or outcome-like columns for the inference secret.
        - Return valid JSON only. No markdown. No explanation.

        JSON format:
        {{
            "linkability_aux_cols": [["col_a", "col_b"],["col_c", "col_d"]],
            "inference_aux_cols": ["col_e", "col_f", "col_g"],
            "secret": "col_h",
            "rationale": "short explanation of why you chose certain assignments."
        }}
    """)
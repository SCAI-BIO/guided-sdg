from textwrap import dedent
from typing import Any

from agentic_synth_data.schemas.models import AgentEval


def evaluation_prompt(
        synthesis_request:dict[str, Any], 
        metrics:dict[str, Any],
    ) -> str:
    return dedent(f"""
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
        - average_jsd <= 0.10 strongly supports a satisfactory assessment.
        - 0.10 < average_jsd <= 0.25 may indicate a need for review.
        - average_jsd > 0.25 may indicate poor fidelity.
        - Columns with JSD > 0.50 are problem columns.

        Normalized correlation difference:
        - Lower is better.
        - Values near 0 indicate preservation of the original correlation structure.
        - Higher values suggest weaker preservation of feature dependencies.

        Discriminator AUC:
        - An AUC near 0.5 is desirable.
        - An AUC closer to 1.0 indicate real and synthetic data are easily distinguishable.

        ### Workflow steps that can be revisited
        | Step | Substep | Description | Agent |
        |------|---------|-------------|-------|
        | 3 | a) | Detect and standardize missing value markers. | |
        |   | b) | Infer data types: Python and semantic. | ✓ |
        |   | c) | Recommend generator: CTGAN, PATEGAN, BayesianNet, PrivBayes. | ✓ |
        |   | d) | Recommend and apply missing value handling strategy. | ✓ |

        Assessment:
        - Average JSD is the primary indicator of marginal distribution fidelity.
        - Correlation difference is the primary indicator of relationship preservation.
        - Discriminator AUC is a supporting indicator and should not be used in isolation.
        - Consider whether the generator preserves both marginal distributions and multivariate relationships.
        - Consider whether the selected generator is appropriate for the dataset characteristics.
        - Consider whether any workflow step should be revisited.
        - Evaluate all metrics together; no single metric should determine the final status.
        - Fidelity metrics are proxies for utility and should be interpreted in terms of expected downstream usefulness.

        Additional considerations:
        - Correlation difference is informative when meaningful feature relationships exist, but may be less informative for datasets with weak correlations.
        - Distribution similarity can be influenced by feature encoding choices, especially for categorical variables.
        - Discriminator models may identify subtle differences that have limited practical impact on downstream utility.
        - Small datasets may naturally lead to weaker fidelity metrics.
        - Mixed categorical/numerical datasets may affect metric interpretation.
        - BayesianNet and PrivBayes may perform differently from GAN-based generators on small datasets and datasets with many categorical variables.

        Problem columns:
        - Problem columns should generally be identified from JSD, while also considering any evidence from correlation preservation or generator behavior.
        - Recommendations should focus on improving the identified problem columns rather than optimizing already well-performing features.
        - Prioritize recommendations that address identified problem columns rather than optimizing already well-performing features.
        - If only a minority of columns are identified as problem columns and overall metrics remain acceptable, prefer "needs_review" over "not_satisfactory".

        Revision guidance:
        - Recommend revisiting a workflow step only when the metrics provide evidence that the step may be contributing to observed weaknesses.
        - High JSD suggests issues with marginal distributions.
        - High JSD concentrated in columns with substantial missing values may indicate inappropriate missing value handling.
        - Poor correlation preservation suggests issues with feature dependencies.
        - High discriminator AUC suggests detectable differences but should be interpreted alongside the other metrics.
        - Explain which metric primarily motivated the recommendation and identify the workflow step most likely to improve it.
        - Suggest the workflow step most likely to improve that issue.
        - Do not recommend revisiting a workflow step unless there is evidence in the metrics that the step may be contributing to the observed weakness.
        - If all metrics are broadly acceptable, prefer preserving the current workflow configuration.

        Allowed status:
        - satisfactory
        - needs_review
        - not_satisfactory

        Status guidance:
        - Assign "not_satisfactory" only when there is substantial evidence of poor fidelity across one or more primary metrics.
        - Isolated weaknesses should generally result in "needs_review" rather than "not_satisfactory".
        - If the status is "satisfactory", avoid recommending revisions unless there is a meaningful weakness in the metrics.

        JSON format:
        {{
            "status": "satisfactory",
            "problem_columns": [],
            "recommendation": "Specific next action. Mention which workflow step should be revisited if needed.",
            "rationale": "Explain why this status was assigned, referring to average JSD, problem columns, generator fit, and any relevant workflow step."
        }}
        """)


def eval_review_message(agent_decision:AgentEval, response:str):
    return f"""
You are answering a question about the synthetic data evaluation notes.

Evaluation results:
{agent_decision.model_dump()}

        User question:
        {response}

        Answer in plain text only.

        Answer the user's question using the evaluation results.
        """)
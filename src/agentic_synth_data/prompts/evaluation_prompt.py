from textwrap import dedent
from typing import Literal

from agentic_synth_data.preference_setup import SynthesisRequest
from agentic_synth_data.schemas.models import AgentEval

from agentic_synth_data.schemas.attack_model import AttackMetrics
from agentic_synth_data.schemas.syndat_model import SyndatMetrics


def evaluation_prompt(
    synthesis_request: SynthesisRequest,
    syndat_metrics: SyndatMetrics,
    attack_metrics: AttackMetrics,
) -> str:
    return dedent(f"""
    Evaluate whether synthetic data quality is satisfactory. If a metric has an output, consider that in your analysis of the dataset's quality.

    Return exactly ONE valid JSON object.
    No markdown.
    No explanation outside of JSON.

    Synthesis request:
    {synthesis_request.model_dump()}

    Generator used:
    {synthesis_request.generator.name}

    Jensen-Shannon Divergence results:
    * {syndat_metrics.jsd}
    * Output for normalized correlation difference: {syndat_metrics.output_messages.jsd}

    Normalized correlation difference:
    * {syndat_metrics.norm_corr_diff}
    * Output for normalized correlation difference: {syndat_metrics.output_messages.norm_corr_diff}

    Discriminator AUC: 
    * {syndat_metrics.discrim_auc}
    * Output for discriminator AUC: {syndat_metrics.output_messages.discrim_auc}

    Evaluate whether the synthetic data is satisfactory.

    Interpretation:
    Jensen-Shannon Divergence: 
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

    Privacy metrics:
    - Values are privacy-risk estimates between 0 and 1.
    - Lower is better.
    - value < 0.10: negligible risk.
    - 0.10 <= value < 0.20: moderate risk; needs review.
    - value >= 0.20: high risk; not satisfactory unless utility is exceptional and privacy was not prioritized.
    - ci[0] and ci[1] are the confidence interval bounds.
    - If the confidence interval is wide, mention uncertainty.
    - Privacy warnings or unreliable estimates should be treated cautiously and mentioned in the rationale.

    ---

    Privacy metric results:
    Singling out risk results:
    * {attack_metrics.singling_out}
    * Output for singling out risk: {attack_metrics.output_messages.singling_out}

    Linkability risk results:
    * {attack_metrics.linkability}
    * Output for singling out risk: {attack_metrics.output_messages.linkability}
    
    Inference risk results:
    * {attack_metrics.inference}
    * Output for singling out risk: {attack_metrics.output_messages.inference}

    Workflow steps that can be revisited:
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
    - Whether privacy risk is low enough based on singling-out, inference, and linkability metrics.
    - Whether privacy was mentioned as an explicit concern.

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
        "recommendation": "Define the next action; name the workflow step to revisit if needed.",
        "rationale": "Justify the status using privacy and fidelity metrics, generator fit, problem features."
    }}
    """)


def eval_review_message(agent_decision:AgentEval, response:str):
    return dedent(f"""
    You are answering a question about the synthetic data evaluation notes.

    Evaluation results:
    {agent_decision.model_dump()}

    User question:
    {response}

    Answer in plain text only.

    Answer the user's question using the evaluation results.
    """)


def attack_column_selection_message(
    semantic_types: dict[str, Literal["Float", "Integer", "Boolean", "Categorical", "Ordinal"]],
) -> str:
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

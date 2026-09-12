from textwrap import dedent
from typing import Literal
from agentic_synth_data.report import Report
from agentic_synth_data.schemas.models import AgentEval
from agentic_synth_data.preference_setup import SynthesisRequest


def revision_message(
        synthesis_request:SynthesisRequest, report:Report, agent_decision:AgentEval,
    ) -> str:
    """Remind agent of all steps taken so far the latest attempt in the workflow and other outputs
    associated with that attempt. This then prompts the agent to recommend a generator update
    (no params at this point).
    """
    if len(report.prep.missing_markers_detected) > 0:
        missing_markers = dedent(f"""
            Missing value markers:
            - Detected (character and count): {report.prep.missing_markers_detected}
            - Replaced as null (character): {report.prep.missing_markers_replaced}
        """)
    else:
        missing_markers = "No missing value markers were detected."

    if synthesis_request.missing_values.mode is not None:
        missing_handling = dedent(f"""
            Missing value handling:
            - Number of missing values before: {report.prep.missing_before}
            - Number of missing values after: {report.prep.missing_after}

            Type of missing value handling applied:
            {synthesis_request.missing_values.model_dump()}
        """)
    else:
        if report.prep.missing_before == 0:
            missing_handling = "- Missing value handling was not applied, no missing values" \
            " present."
        if synthesis_request.generator.name in ["PrivBayes", "BayesianNet"]:
            missing_handling = "- Missing value handling was not applied, the generator used" \
            " allows for missing values."
    return dedent(f"""
    The workflow should be revised. For now, only start with generator selection.
    Input dataframe shape: {report.input.original_shape}
    Prepared dataframe shape (used as an input for generating synthetic data): {report.input.prepared_shape}
    - Note: shape will only be lower than input if missing values are not handled.
    Synthetic dataframe shape: {report.gen.synth_shape}

    {missing_markers}

    The detected dataframe dtypes: {report.prep.dtypes}

    The detected dataframe semantic types: {report.prep.stypes}

    Correlations (numerical values only):
    {report.prep.profile}

    Chosen generator: {synthesis_request.generator.name}

    {missing_handling}

    Syndat metrics:
    Average jsd: {round(report.review.jsd['jsd_AVERAGE'], 4)}
    Normalized correlation difference: {round(report.review.norm_corr_diff, 4)}
    Discriminator AUC: {round(report.review.discrim_auc, 4)}

    Your recent evaluation after generation:
    - Status: {agent_decision.status}
    - Recommendation: {agent_decision.recommendation}
    - Rationale: {agent_decision.rationale}

    Allowed revision scope:
    - You may only suggest changes to generator selection.
    - Do not suggest missing value handling changes.
    - Do not suggest dtype changes.
    - Do not suggest semantic type changes.
    - Do not suggest sample size changes.
    - Do not suggest privacy mode changes.

    Available generators:
    - CTGAN
    - PATEGAN
    - BayesianNet
    - PrivBayes

    Return valid JSON only with this structure:
    {{
    "options": [
        {{
        "id": "option_1",
        "generator_name": "CTGAN",
        "parameters": {{}},
        "reason": "short reason based on evaluation metrics"
        }}
    ]
    }}

    Give 1 to 3 options.
    """)


def parameter_revision_message(selected_mechanism:Literal["PATEGAN", "CTGAN", "PrivBayes", "BayesianNet"]) -> str:
    return dedent(f"""
    The following generator has been selected {selected_mechanism}.

    Infer whether its default parameters should be updated:

    Default parameters:
    - PATEGAN: eps=1, delta=1e-5, infer_ranges=False, num_teachers=10, n_iters=100, batch_size=128, learning_rate=1e-4, multiprocess=False
    - BayesianNet: histogram_bins=10, degree=1, infer_ranges=False, multiprocess=True, seed=None
    - PrivBayes: histogram_bins=10, degree=1, epsilon=.1, infer_ranges=False, multiprocess=True, seed=None
    - CTGAN: embedding_dim=128, gen_dim=(256, 256), dis_dim=(256, 256), l2scale=1e-6, batch_size=500, epochs=300, multiprocess=False

    Parameters that can be changed for each model:
    PATEGAN:
    - eps: integer between 0 and 10
    PrivBayes:
    - epsilon: float between 0.0 and 1.0
    CTGAN: No parameter updates allowed.
    BayesianNet: No parameter updates allowed.

    Return valid JSON only with this structure:
    {{"epochs": 500}}
    """)


# NOTE: have not included threshold as an option here (not stable with the current setup)
def missing_revision_message(synthesis_request:SynthesisRequest):
    return dedent(f"""
    The workflow should be revised. A generator was selected that does not allow for missing values.

    The previous missing value handling strategy was:
    {synthesis_request.missing_values.model_dump()}

    Pay attention to the number of missing values present as well as your previous assessments and 
    potential conversations to inform your suggestion. Decide whether any changes are recommended.

    Allowed modes:
    - remove
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
    - If the mode value has a key of the same name (impute), all entries within that key must be filled.
    - If there are no missing values, recommend mode "remove" and explain that no rows will be removed.
    - If few rows have missing values, recommend "remove".
    - If missing values are spread across many rows, recommend "impute".
    - Do not suggest invalid options.

    IMPORTANT: If mode is "impute", BOTH categorical AND numerical strategies CANNOT be empty.
    - If mode is not "impute", categorical and numerical strategy must be empty.

    If a change is suggested that does not make sense for the dataset's features: 
    - In "warning" specify the choice(s) that are not possible, or do not make sense for the dataset.
    - In "reason" briefly specify what would be a better fit and why.

    Return valid JSON only with this structure:
    {{
    "options": [
        {{
        "id": "option_1",
        "mode": "remove",
        "imputation_method": null,
        "categorical_strategy": "most_frequent",
        "numerical_strategy": "mean",
        "reason": "Brief explanation framed by the dataset's information."
        }}
    ]
    }}

    Give 1 to 3 options.
    """)

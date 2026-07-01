from textwrap import dedent
from typing import Any

from agentic_synth_data.preference_setup import SynthesisRequest


# NOTE: missingness per column could also be input here
def generator_selection_message(
    synthesis_request: SynthesisRequest, 
    semantic_types: dict[str, str],
    profile:dict[str, Any],
    ) -> str:
    return dedent(f"""
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
    {synthesis_request.model_dump()}

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

    Selection rules:
    - If synthesis_request["generator"]["name"] is not "auto":
      - Use the requested generator.
      - Do not override the user's choice.
      - If the generator appears unsuitable, explain the concern in "warning".
    - Otherwise (general preferences):
      - If privacy mode is differentially_private:
        - mostly categorical data: PrivBayes
        - mostly numerical or mixed data: PATEGAN
      - If privacy mode is standard:
        - mostly categorical data: BayesianNet
        - mostly numerical data: CTGAN
        - mixed data: CTGAN

    These are not strict requirements. If the dataset characteristics strongly suggest another generator is more appropriate, explain the reasoning in the rationale.

    Generator selection principles:
    - There is no universally best generator.
    - Generator suitability depends on dataset size, feature types, cardinality, and feature dependencies.
    - Do not assume that differential privacy implies better utility or fidelity.
    - Do not assume that non-private generators are always preferable.
    - Differential privacy is one consideration among many and may reduce utility in some datasets.
    - Select the generator that is expected to provide the best balance between privacy requirements, fidelity, and utility for this dataset.
    - Use the dataset characteristics provided above rather than relying solely on generator family assumptions.

    Dataset considerations:
    - Small datasets may limit the ability of complex generators to learn realistic distributions.
    - Datasets with many categorical columns may favor generators designed for discrete relationships.
    - Datasets with strong feature dependencies may benefit from generators that better model multivariate structure.
    - Datasets with high-cardinality categorical variables may present additional modeling challenges.

    Warning rules:
    - If the selected generator does not fit the dataset well, include a warning.
      - In "warning" specify the choice(s) that are not possible, or do not make sense for the dataset.
      - In "rationale" briefly specify what would be a better fit and why.
    - Otherwise, use an empty warning list.

    Rationale requirements:
    - Reference specific dataset characteristics.
    - Explain why the selected generator is more appropriate than the most likely alternative.
    - Do not justify a choice solely based on privacy mode.
    - Avoid generic statements such as "handles mixed data well" or "good for tabular data".
    - Focus on expected fidelity, utility, privacy requirements, and dataset characteristics.
      
    Description rules:
    - Do not be repetitive.
    - Do not refer to selection rules.
    - Be specific in the context of the dataset.
    - Pros and cons should describe tradeoffs of the selected generator for this dataset, not generic generator characteristics.

    JSON format:
    {{
      "generator": "generator_name",
      "pros": "Briefly describe the BENEFITS given the dataset's information.",
      "cons": "Briefly describe the DRAWBACKS given the dataset's information.",
      "rationale": "Briefly describe the choice in the context of the dataset.",
      "warning": []
    }}
    """)


def generator_review_message(selected_mechanism):
    return dedent(f"""
      The following generator is being considered:
      {selected_mechanism}

      Give pros, cons, rationale, and warnings for this selected generator.
      Do not automatically change the generator unless there is a strong reason.
      """)
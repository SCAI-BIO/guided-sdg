from pathlib import Path
from typing import Optional, Any
import pprint

import pandas as pd
import numpy as np

from langgraph.graph.state import CompiledStateGraph
import pyinputplus as pyip
from syndat.metrics import jensen_shannon_distance, normalized_correlation_difference, discriminator_auc

from agentic_synth_data.schemas.models import AgentEval
from agentic_synth_data.utils import format_response, extract_output, parse_json
from agentic_synth_data.prompts.evaluation_prompt import eval_review_message, evaluation_prompt
from agentic_synth_data.preference_setup import SynthesisRequest
from agentic_synth_data.report import Report


def calculate_metrics(real_df:pd.DataFrame, synthetic_df:pd.DataFrame) -> dict[str, Any]:
    """Calculate metrics comparing the original dataframe (optionally, with missing value markers removed)
    with the synthetic dataframe.

    Calculates SYNDAT metrics:
    - Jenson-Shannon Distance
    - Normalized Correlation Difference
    - Discriminator AUC

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synthetic_df (pd.DataFrame): The synthetically generated dataframe.

    Returns:
        dict: The decision plan based on the agent's assessment.
    """
    jsd_results = jensen_shannon_distance(real=real_df, synthetic=synthetic_df)
    jsd_results["jsd_AVERAGE"] = np.nanmean(list(jsd_results.values())) # nanmean just in case jsd converts a column to nan
    
    print("\nJenson-Shannon Distances:")
    for col, jsd_score in jsd_results.items():
        print(col, round(jsd_score, 4))
    
    norm_corr_diff = normalized_correlation_difference(real=real_df, synthetic=synthetic_df, ignore_cat=True)
    print(f"\nNormalized correlation difference: {round(norm_corr_diff, 4)}")

    discrim_auc = discriminator_auc(real=real_df, synthetic=synthetic_df)
    print(f"\nDiscriminator AUC: {round(discrim_auc, 4)}")

    return {
        "jsd": jsd_results,
        "norm_corr_diff": norm_corr_diff,
        "discrim_auc": discrim_auc,
    }


def evaluation(
        synthesis_request:SynthesisRequest,
        metrics:dict[str, Any],
        agent:Optional[CompiledStateGraph] | None = None, 
        config:dict | None = None
    ) -> AgentEval:
    """The agent evaluates the synthetic dataset using calculated metrics.

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synthetic_df (pd.DataFrame): The synthetically generated dataframe.
        synthesis_request (SynthesisRequest): Dictionary of the user's preferences. This is used as a
                                record of the processing done to generate the synthetic data.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to evaluate the
                                synthetic data based on SYNDAT metrics. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation. Defaults to None.

    Returns:
        AgentEval: The decision plan based on the agent's assessment.
    """
    request_dict = synthesis_request.model_dump()
    prompt = evaluation_prompt(synthesis_request=request_dict, metrics=metrics,)
    
    
    print("\nAgent is evaluating the results...")
    
    raw_response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        },
        config=config,
    )

    plan = extract_output(raw_response)

    eval_plan = parse_json(plan, AgentEval, agent, config)
    
    print("\nEvaluation:")
    print(f"- Status: {eval_plan.status}")
    if len(eval_plan.problem_columns) > 0:
        print(f"- Problem columns: {eval_plan.problem_columns}")
    print(f"- Recommendation:")
    format_response(eval_plan.recommendation)
    print("- Rationale:")
    format_response(eval_plan.rationale)
    return eval_plan


def evaluation_review(
        agent_decision:AgentEval, 
        agent:Optional[CompiledStateGraph] | None = None, 
        config:dict | None = None
        ) -> str:
    """User asks questions about agent's assessment/discusses details to focus on.

    Args:
        agent_decision (AgentEval): Agent's evaluation decision.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to review the evaluation. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation.. Defaults to None.

    Returns:
        str: Whether the workflow should finish or be revised.
    """
    confirm = pyip.inputYesNo("\nDo you have questions about the evaluation results?\n")
    
    while confirm == "yes":
        response = pyip.inputStr("\nEnter your question/concern:\n")
        print("\nAgent is preparing its response...")
        prompt = eval_review_message(eval_decision, response)
        raw_response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            },
            config=config,
        )

        format_response(extract_output(raw_response))

        confirm = pyip.inputYesNo("\nDo you have another question?\n")

    print(f"\nWorkflow result status: {agent_decision.status}")
    choice = pyip.inputMenu(["finish", "revise"], "\nWould you like to finish or revise the workflow?\n", numbered=True)
    
    return choice


def write_report(report: Report, synthesis_request:SynthesisRequest, data_path: str, version_id:int) -> None:
    """Write a workflow report based on decisions made during the synthetic data generation process.

    Args:
        report (Report): Dictionary containing information gathered throughout the workflow steps.
        synthesis_request (SynthesisRequest): Dictionary with the user's preferences.
        data_path (str): Initial path of the data (used for naming the report).
        version_id (int): Denotes the version of the workflow attempt.
    """
    request_dict = synthesis_request.model_dump()
    data_name = Path(data_path).stem
    output_filename = f"report_{data_name}_v{version_id}.md"
    output_path = Path("../output") / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# SYNTHETIC DATA GENERATION REPORT\n")

        f.write("## Input Dataset:\n")
        f.write(f"- Path: {report.input.path}\n")
        f.write(f"- Original shape: {report.input.original_shape}\n")
        f.write(f"- Prepared shape: {report.input.prepared_shape}\n\n")

        f.write("## Preprocessing:\n")
        f.write("Missing Value Detection\n")
        f.write("Detected markers:\n")
        f.write("```python\n")
        pprint.pprint(report.prep.missing_markers_detected, stream=f, sort_dicts=False,)

        f.write("\n Replaced markers:\n")
        f.write("```python\n")
        pprint.pprint(report.prep.missing_markers_replaced, stream=f, sort_dicts=False,)
        f.write("```\n")

        f.write("\n\n### Python data types\n")
        f.write("```python\n")
        pprint.pprint(report.prep.dtypes, stream=f, sort_dicts=False)
        f.write("```\n")

        f.write("\n\n### Semantic data types\n")
        f.write("```python\n")
        pprint.pprint(report.prep.stypes, stream=f, sort_dicts=False)
        f.write("```\n")

        f.write("\n\n### Missing Value Handling\n")
        f.write("```python\n")
        pprint.pprint(report.prep.missing_before, stream=f, sort_dicts=False,)
        pprint.pprint(report.prep.missing_after, stream=f, sort_dicts=False,)
        f.write("```\n")

        f.write("\n\n### Generation:\n")
        f.write(f"Chosen generator: {request_dict['generator']}\n")
        f.write(f"Prepared dataset: {report.gen.prep_path}\n")
        f.write(f"Synthetic dataset: {report.gen.synth_path}\n")
        f.write(f"Synthetic shape: {report.gen.synth_shape}\n")

        f.write("\n\n### Evaluation\n")
        f.write("```python\n")
        pprint.pprint(report.review.jsd, stream=f, sort_dicts=False)
        f.write("```\n")

        f.write("\nConclusion\n")
        f.write("```python\n")
        pprint.pprint(report.review.recommendation, stream=f, sort_dicts=False,)
        f.write("```\n")
from pathlib import Path
from typing import Optional, Any, Literal
import pprint

from anonymeter.stats.confidence import PrivacyRisk
from anonymeter.evaluators import InferenceEvaluator, LinkabilityEvaluator, SinglingOutEvaluator
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from langgraph.graph.state import CompiledStateGraph
import pyinputplus as pyip
from syndat.metrics import jensen_shannon_distance, normalized_correlation_difference, discriminator_auc

from agentic_synth_data.utils import format_response, extract_output, parse_json
from agentic_synth_data.prompts.evaluation_prompt import attack_column_selection_message, eval_review_message, evaluation_prompt
from agentic_synth_data.preference_setup import SynthesisRequest


class AgentEval(BaseModel):
    status: Literal["satisfactory", "needs_review", "not_satisfactory"]
    problem_columns: list[str] = Field(default_factory=list)
    recommendation: str
    rationale: str


class AttackColumns(BaseModel):
    linkability_aux_cols:list[list[str]] = Field(min_length=2, max_length=2)
    inference_aux_cols:list[str] = Field(min_length=1) # possible TODO: Add additional constraints
    secret:str
    rationale:str


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


def calculate_attack_metrics(
    real_df:pd.DataFrame,
    synthetic_df:pd.DataFrame,
    semantic_types:dict[str, str],
    agent:Optional[CompiledStateGraph] | None = None, 
    config:dict | None = None
    ) -> dict[str, PrivacyRisk]:
    print("\nEvaluating singling out risk...")
    singling_out = SinglingOutEvaluator(ori=real_df, syn=synthetic_df, max_attempts=1_000)
    singling_out.evaluate()
    singling_out_risk = singling_out.risk()
    print(f"\n=> Singling out risk:\n  - Value: {round(singling_out_risk.value*100, 3)}%" \
        f"\n  - Confidence interval between: {round(singling_out_risk.ci[0]*100,3)}%" \
        f" and {round(singling_out_risk.ci[1]*100,3)}%")

    prompt = attack_column_selection_message(semantic_types=semantic_types)
    print("\nAgent is selecting columns for the following privacy metrics to be assessed:" \
        "\n- Inference risk (short explanation)\n- Linkability risk (short explanation)")

    raw_response = agent.invoke({"messages": [{"role": "user", "content": prompt,}]}, config=config,)
    raw_response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt,}]}, config=config,
        )
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

    selected_cols = parse_json(plan, AttackColumns, agent, config)

    print(f"\nSelected columns:\n- Linkability:\n  - Auxiliary columns: " \
        f"{selected_cols.linkability_aux_cols}\n- Inference:\n  - Auxiliary" \
        f" columns: {selected_cols.inference_aux_cols}\n  - Secret " \
        f"column: {selected_cols.secret}\n\nRationale:")
    format_response(selected_cols.rationale)

    print("\nEvaluating inference risk...")
    inference = InferenceEvaluator(
        ori=real_df,
        syn=synthetic_df,
        aux_cols=selected_cols.inference_aux_cols,
        secret=selected_cols.secret,
        n_attacks=min(500, len(real_df))
        )
    inference.evaluate()
    inference_risk = inference.risk()
    print(f"\n=> Inference out risk:\n  - Value: {round(inference_risk.value*100, 3)}%" \
        f"\n  - Confidence interval: between {round(inference_risk.ci[0]*100,3)}%" \
        f" and {round(inference_risk.ci[1]*100,3)}%")

    print("\nEvaluating linkability risk...")
    linkability = LinkabilityEvaluator(
        ori=real_df,
        syn=synthetic_df,
        aux_cols=selected_cols.linkability_aux_cols,
        n_attacks=min(500, len(real_df))
        )
    linkability.evaluate()
    linkability_risk = linkability.risk()
    print(f"\n=> Linkability risk:\n  - Value: {round(linkability_risk.value*100, 3)}%" \
        f"\n  - Confidence interval: between {round(linkability_risk.ci[0]*100,3)}%" \
        f" and {round(linkability_risk.ci[1]*100,3)}%")

    return {
        "singling_out_risk": singling_out_risk,
        "inference_risk": inference_risk,
        "linkability_risk": linkability_risk,
    }


def evaluation(
        synthesis_request:SynthesisRequest,
        metrics:dict[str, Any],
        attack_metrics:dict[str, Any],
        agent:Optional[CompiledStateGraph] | None = None, 
        config:dict | None = None
        ) -> dict:
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
        dict: The decision plan based on the agent's assessment.
    """
    request_dict = synthesis_request.model_dump()
    prompt = evaluation_prompt(synthesis_request=request_dict, metrics=metrics, attack_metrics=attack_metrics)
    
    
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
    
    decision = {
        "status": eval_plan.status,
        "problem_columns": eval_plan.problem_columns,
        "recommendation": eval_plan.recommendation,
        "rationale": eval_plan.rationale,
    }
    return decision


def evaluation_review(
        eval_decision:dict[str, Any], 
        agent:Optional[CompiledStateGraph] | None = None, 
        config:dict | None = None
        ) -> str:
    """User asks questions about agent's assessment/discusses details to focus on.

    Args:
        eval_decision (dict[str, Any]): Agent's evaluation decision (from evaluation).
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to review the evaluation. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation.. Defaults to None.

    Returns:
        str: Whether the workflow should finish or be revised.
    """
    confirm = pyip.inputYesNo("\nDo you have questions about the evaluation results?\n")
    
    while confirm == "yes":
        response = pyip.inputStr("\nEnter your question/concern:\n")
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

    print(f"\nWorkflow result status: {eval_decision['status']}")
    choice = pyip.inputMenu(["finish", "revise"], "\nWould you like to finish or revise the workflow?\n", numbered=True)
    
    return choice


def write_report(report: dict[str, dict[str, Any]], synthesis_request:SynthesisRequest, data_path: str, version_id:int) -> None:
    """Write a workflow report based on decisions made during the synthetic data generation process.

    Args:
        report (dict[str, dict[str, Any]]): Dictionary containing information gathered throughout the workflow steps.
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
        f.write(f"- Path: {report['input']['path']}\n")
        f.write(f"- Original shape: {report['input']['original_shape']}\n")
        f.write(f"- Prepared shape: {report['input']['prepared_shape']}\n\n")

        f.write("## Preprocessing:\n")
        f.write("Missing Value Detection\n")
        f.write("Detected markers:\n")
        f.write("```python\n")
        pprint.pprint(report["prep"]["missing_markers"]["detected"], stream=f, sort_dicts=False,)

        f.write("\n Replaced markers:\n")
        f.write("```python\n")
        pprint.pprint(report["prep"]["missing_markers"]["replaced"], stream=f, sort_dicts=False,)
        f.write("```\n")

        f.write("\n\n### Python data types\n")
        f.write("```python\n")
        pprint.pprint(report["prep"]["dtypes"], stream=f, sort_dicts=False)
        f.write("```\n")

        f.write("\n\n### Semantic data types\n")
        f.write("```python\n")
        pprint.pprint(report["prep"]["stypes"], stream=f, sort_dicts=False)
        f.write("```\n")

        f.write("\n\n### Missing Value Handling\n")
        f.write("```python\n")
        pprint.pprint(report["prep"]["missing_handling"], stream=f, sort_dicts=False,)
        f.write("```\n")

        f.write("\n\n### Generation:\n")
        f.write(f"Chosen generator: {request_dict['generator']}\n")
        f.write(f"Prepared dataset: {report['gen']['prep_path']}\n")
        f.write(f"Synthetic dataset: {report['gen']['synth_path']}\n")
        f.write(f"Synthetic shape: {report['gen']['synth_shape']}\n")

        f.write("\n\n### Evaluation\n")
        f.write("```python\n")
        pprint.pprint(report["review"]["jsd"], stream=f, sort_dicts=False)
        f.write("```\n")

        f.write("\nConclusion\n")
        f.write("```python\n")
        pprint.pprint(report["review"]["recommendation"], stream=f, sort_dicts=False,)
        f.write("```\n")
import json
import pprint
from typing import Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path

from syndat.metrics import jensen_shannon_distance, normalized_correlation_difference, discriminator_auc

from langgraph.graph.state import CompiledStateGraph

from agentic_synth_data.utils import format_response
from agentic_synth_data.prompts.evaluation_prompt import evaluation_prompt


# possible TODO: hardcoding a threshold to limit the chance that the agent hallucinates information.
# TODO: add docstring (finish typings)
def evaluation(
        real_df:pd.DataFrame, 
        synthetic_df:pd.DataFrame, 
        synthesis_request:dict[str, Any],
        agent:Optional[CompiledStateGraph] | None = None, 
        config:dict | None = None
        ) -> dict:
    generator_used = synthesis_request["generator"]
    jsd_results = jensen_shannon_distance(real=real_df, synthetic=synthetic_df)
    jsd_results["jsd_AVERAGE"] = np.nanmean(list(jsd_results.values())) # nanmean just in case jsd converts a column to nan
    print("Jenson-Shannon Distances:")
    pprint.pprint(jsd_results)
    print()

    norm_corr_diff = normalized_correlation_difference(real=real_df, synthetic=synthetic_df)
    print(f"\nNormalized correlation difference: {norm_corr_diff}")

    discrim_auc = discriminator_auc(real=real_df, synthetic=synthetic_df)
    print(f"\nDiscriminator AUC: {discrim_auc}")

    prompt = evaluation_prompt(
        synthesis_request=synthesis_request, 
        generator_used=generator_used,
        jsd_results=jsd_results,
        norm_corr_diff=norm_corr_diff,
        discrim_auc=discrim_auc,
        )
    
    print("\nAgent is evaluating the results...")
    
    plan = agent.invoke(
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

    plan = plan["messages"][-1].content.split("</think>")[1].strip()
    eval_plan = json.loads(plan)

    print("\nEvaluation:")
    print(f"- Status: {eval_plan['status']}")
    print(f"- Problem columns: {eval_plan['problem_columns']}")
    print(f"- Recommendation:")
    format_response(eval_plan["recommendation"])
    print("- Rationale:")
    format_response(eval_plan["rationale"])

    decision = {
    "status": eval_plan['status'],
    "jsd": jsd_results,
    "norm_corr_diff": norm_corr_diff,
    "discrim_auc": discrim_auc,
    "problem_columns": eval_plan['problem_columns'],
    "recommendation": eval_plan["recommendation"],
    "rationale": eval_plan["rationale"],
    }
    # TODO:implement greater communication of results.
    return decision

# TODO make a tool for the agent to rerun synthetic data generation method can be expanded to additional steps later.#
    # - Should allow for user confirmation of the agent's assessments.
    # - Should allow for the display and editing of steps
    # - Should finally redirect to previous steps (not sure if confirmations with user are necessary or not when rerun)
    # NOTE: the report that is generated should be updated stating which step was rerun and update the final report with 
        # the appended additions to the workflow. 
        # - Possibly then allow for the agent to make a comparison between both stages and agent suggests which version to keep
        # (this would likely be good to update the final report with)

def write_report(report: dict[str, dict[str, Any]], synthesis_request:dict[str, Any], data_path: str) -> None:
    """Write a workflow report based on decisions made during the synthetic data generation process.

    Args:
        report (dict[str, dict[str, Any]]): Dictionary containing information gathered throughout the workflow steps.
        synthesis_request (dict[str, Any]): Dictionary with the user's preferences.
        data_path (str): Initial path of the data (used for naming the report).
    """
    data_name = Path(data_path).stem
    output_filename = f"report_{data_name}.txt"
    output_path = Path("../output") / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SYNTHETIC DATA GENERATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Input Dataset:\n")
        f.write(f"Path: {report['input']['path']}\n")
        f.write(f"Original shape: {report['input']['original_shape']}\n")
        f.write(f"Prepared shape: {report['input']['prepared_shape']}\n\n")

        f.write("Preprocessing:\n")
        f.write("Missing Value Detection\n")
        f.write(f"Detected markers:\n")
        pprint.pprint(report["prep"]["missing_markers"]["detected"], stream=f, sort_dicts=False,)

        f.write("\nReplaced markers:\n")
        pprint.pprint(report["prep"]["missing_markers"]["replaced"], stream=f, sort_dicts=False,)

        f.write("\n\nPython data types\n")
        pprint.pprint(report["prep"]["dtypes"], stream=f, sort_dicts=False)

        f.write("\n\nSemantic data types\n")
        pprint.pprint(report["prep"]["stypes"], stream=f, sort_dicts=False)

        f.write("\n\nMissing Value Handling\n")
        pprint.pprint(report["prep"]["missing_handling"], stream=f, sort_dicts=False,)

        f.write("\n\nGeneration:\n")
        f.write(f"Chosen generator: {synthesis_request["generator"]}\n")
        f.write(f"Prepared dataset: {report['gen']['prep_path']}\n")
        f.write(f"Synthetic dataset: {report['gen']['synth_path']}\n")
        f.write(f"Synthetic shape: {report['gen']['synth_shape']}\n")

        f.write("\n\nEvaluation\n")
        pprint.pprint(report["eval"]["jsd"], stream=f, sort_dicts=False)

        f.write("\nConclusion\n")
        pprint.pprint(report["eval"]["recommendation"], stream=f, sort_dicts=False,)
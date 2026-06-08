from pathlib import Path
from typing import Optional, Any
import pprint
import json

import pandas as pd
import numpy as np
from langgraph.graph.state import CompiledStateGraph
from syndat.metrics import jensen_shannon_distance, normalized_correlation_difference, discriminator_auc

from agentic_synth_data.utils import format_response, extract_output
from agentic_synth_data.prompts.evaluation_prompt import evaluation_prompt


# possible TODO: hardcoding a threshold to limit the chance that the agent hallucinates information.
def evaluation(
        real_df:pd.DataFrame, 
        synthetic_df:pd.DataFrame, 
        synthesis_request:dict[str, Any],
        agent:Optional[CompiledStateGraph] | None = None, 
        config:dict | None = None
        ) -> dict:
    """SYNDAT metrics are calculated (Jenson-Shannon Distance, Normalized Correlation Difference,
    Discriminator AUC) for the synthetic dataset, which the agent then evaluates and determines
    whether or not additional steps are needed, and possible alternative choices that could be
    pursued to improve results (if necessary). 

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synthetic_df (pd.DataFrame): The synthetically generated dataframe.
        synthesis_request (dict[str, Any]): Dictionary of the user's preferences. This is used as a
                                record of the processing done to generate the synthetic data.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to evaluate the
                                synthetic data based on SYNDAT metrics. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation. Defaults to None.

    Returns:
        dict: _description_
    """
    generator_used = synthesis_request["generator"]
    jsd_results = jensen_shannon_distance(real=real_df, synthetic=synthetic_df)
    jsd_results["jsd_AVERAGE"] = np.nanmean(list(jsd_results.values())) # nanmean just in case jsd converts a column to nan
    print("Jenson-Shannon Distances:")
    for col, jsd_score in jsd_results.items():
        print(col, jsd_score)
    print()

    norm_corr_diff = normalized_correlation_difference(real=real_df, synthetic=synthetic_df, ignore_cat=True)
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


def write_report(report: dict[str, dict[str, Any]], synthesis_request:dict[str, Any], data_path: str) -> None:
    """Write a workflow report based on decisions made during the synthetic data generation process.

    Args:
        report (dict[str, dict[str, Any]]): Dictionary containing information gathered throughout the workflow steps.
        synthesis_request (dict[str, Any]): Dictionary with the user's preferences.
        data_path (str): Initial path of the data (used for naming the report).
    """
    data_name = Path(data_path).stem
    output_filename = f"report_{data_name}.md"
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
        f.write(f"Chosen generator: {synthesis_request["generator"]}\n")
        f.write(f"Prepared dataset: {report['gen']['prep_path']}\n")
        f.write(f"Synthetic dataset: {report['gen']['synth_path']}\n")
        f.write(f"Synthetic shape: {report['gen']['synth_shape']}\n")

        f.write("\n\n### Evaluation\n")
        f.write("```python\n")
        pprint.pprint(report["eval"]["jsd"], stream=f, sort_dicts=False)
        f.write("```\n")

        f.write("\nConclusion\n")
        f.write("```python\n")
        pprint.pprint(report["eval"]["recommendation"], stream=f, sort_dicts=False,)
        f.write("```\n")
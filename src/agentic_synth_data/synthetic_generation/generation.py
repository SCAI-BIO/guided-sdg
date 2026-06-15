from pathlib import Path
import subprocess as sp
from typing import Literal, Any

import pandas as pd
import pyinputplus as pyip
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agentic_synth_data.prompts.generator_selection_prompt import generator_selection_message, generator_review_message
from agentic_synth_data.utils import format_response, extract_output, parse_json
from agentic_synth_data.preference_setup import SynthesisRequest


class SelectGenerator(BaseModel):
    generator: Literal["PrivBayes", "CTGAN", "PATEGAN", "BayesianNet"]
    pros: str
    cons: str
    rationale: str
    warning: list[str]


# TODO: If run_generator fails, for example, plug the error message into the agent and try to
# have the agent repair the issue.
def run_generator(data_path:str, clean_csv_path:str, synthesis_request:SynthesisRequest, version_id:int) -> str:
    """Run the selected data generator based on user preferences and agent recommendation.

    Args:
        data_path (str): Initial path of the data (used to obtain original sample size if 
                        same_as_original is selected).
        clean_csv_path (str): Path of the prepared data (ends with "_prepared.csv"). Used as
                        a basis to generate the synthetic data. This dataset may include the 
                        following changes:
                        - Missing value marker replacement with null
                        - dtype handling
                        - Missing value handling
        synthesis_request (SynthesisRequest): Dictionary of the user's preferences, with the 
                        decided generator.

    Returns:
        str: Path of the synthetic dataset.                                  
    """
    mechanism = synthesis_request.generator.name
    # Set sample size
    if synthesis_request.sample_size.mode == "fixed_value":
        sample_size = synthesis_request.sample_size.fixed_value
    else:
        sample_size = pd.read_csv(data_path).shape[0]

    # Set cleaned data and output paths
    data_name = Path(clean_csv_path).stem
    output_filename = f"synthetic_{data_name}_{mechanism.lower()}_v{version_id}.csv"
    host_output_path = Path("../output") / output_filename
    container_output_path = f"/workspace/output/{output_filename}"
    
    print()
    print("-"*25+f"Running {mechanism}"+"-"*25)

    command = f"""
    podman run -it --rm \
        --entrypoint /bin/bash -v "$PWD/..:/workspace:Z" \
        docker.io/springepfl/synthetic-data:latest \
        -c "cd /synthetic_data_release && python3 executables/generate_synthetic_dataset.py \
        -D /workspace/data/{data_name} \
        -M {mechanism} \
        -O {container_output_path} \
        -N {sample_size}"
        """
    process = sp.Popen(
        command,
        shell=True,
        stdout=sp.PIPE,
        stderr=sp.STDOUT,
        text=True
        )

    for line in process.stdout:
        print(line, end="")

    process.wait()
    
    # If process fails for a specific generator
    if process.returncode !=0:
        raise RuntimeError(f"{mechanism} failed.")

    print(f"\n=> {mechanism}: {sample_size} synthetic samples successfully generated.")
    return str(host_output_path)


def generator_profile(dtype_df:pd.DataFrame) -> dict[str, Any]:
    """Prior to selecting the generator, view correlations and summary statistics 
    of numeric columns, which could help to inform data generator choice.

    Args:
        dtype_df (pd.DataFrame): The dataframe after dtype handling (dtype_handled_df).

    Returns:
        dict[str, Any]: A dictionary with the following information:
            - The shape of the dataframe (no row removal up to this point).
            - The number of unique values per column.
            - Numeric columns:
                - Summary statistics (including skew).
                - Correlation.
                - Correlation method.
    """
    # TODO: Add a warning about summary statistics for non local models.
    # TODO: Add more metrics for categorical columns.
    df = dtype_df.copy()
    col_profiles = {}
    shape = df.shape

    for col in df.columns:
        col_profiles[col] = {
            "n_unique": df[col].nunique(dropna=False),
        }
    
    num_df = df.select_dtypes(include="number")

    if num_df.empty:  
        return {
            "shape": shape,
            "col_nunique": col_profiles,
            "num_summary": None,
            "num_corr": None,
            "corr_method": None,
        }

    # Select summary statistics (including skew)
    num_summary = num_df.describe()
    num_summary.loc["skew"] = num_df.skew().values
    #stats.loc["kurt"] = num_df.kurtosis().values # from initial, not fully necessary

    print("Numeric value summary statistics:")
    print(num_summary)

    # Select correlations (Pearson or Spearman)
    # NOTE: could decide on the percentage of highly skewed columns
    if num_df.skew().abs().max() > 2:
        corr_method = "spearman"
    else:
        corr_method = "pearson"

    num_corr = num_df.corr(method=corr_method, numeric_only=True)

    print(f"\nCorrelation ({corr_method}):")
    print(num_corr)

    profile = {
        "shape": shape,
        "col_nunique": col_profiles,
        "num_summary": num_summary,
        "num_corr": num_corr,
        "corr_method": corr_method,
    }
    return profile


# TODO: Expand to parameter selection.
def select_generator(
        synthesis_request:SynthesisRequest, 
        semantic_types:dict[str, str],
        profile:dict[str, Any],
        agent:CompiledStateGraph, 
        config:dict[str, dict[str, Any]],
    ) -> SynthesisRequest:
    """Based on the user's preferences and information about the dataset, the agent 
    selects a data generator mechanism.

    Args:
        synthesis_request (SynthesisRequest): Dictionary of the user's preferences.
        semantic_types (dict[str, str]): The semantic dtypes for each column in the dataset.
        agent (CompiledStateGraph): LangGraph agent used to recommend a mechanism/generator.
        config (dict[str, dict[str, Any]]: Configuration passed to the agent during invocation.

    Returns:
        SynthesisRequest: Synthesis request with updated generator information.                       
    """
    feedback_hist = []
    selected_mechanism = None
    choice_confirmed = False

    print()
    format_response("Agent is coming up with a generator suggestion based on your preferences, the dataset's semantic types, the cardinality of the columns, and potential summary statistics and correlations...") 

    while not choice_confirmed:
        request_dict = synthesis_request.model_dump()
        message = generator_selection_message(
            synthesis_request=request_dict, 
            semantic_types=semantic_types,
            profile=profile
        )

        if selected_mechanism is not None:
            message += generator_review_message(selected_mechanism)

        if feedback_hist:
            message += "\n\nUser feedback so far:\n" + "\n".join(feedback_hist)

        raw_response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
            },
            config=config,
        )

        plan = extract_output(raw_message=raw_response)

        generation_plan = parse_json(plan=plan, model_cls=SelectGenerator, agent=agent, config=config) # TODO: check more extensively

        print("\nSuggestion:")
        print(f"Data generator: {generation_plan.generator}")

        print("\nPros:")
        format_response(generation_plan.pros)

        print("\nCons:")
        format_response(generation_plan.cons)

        print("\nRationale:")
        format_response(generation_plan.rationale)

        if len(generation_plan.warning) > 0:
            print("\nWarning:")
            format_response(generation_plan.warning[0])

        print("\nDo you approve using the suggested generator?\n")
        decision = pyip.inputMenu(["approve", "choose another", "discuss",], numbered=True,)

        if decision == "approve":
            synthesis_request.generator.name = generation_plan.generator
            choice_confirmed = True
            print(f"\n=> {generation_plan.generator} has been approved.")

        elif decision == "choose another":
            selected_mechanism = pyip.inputMenu(
                ["PATEGAN", "CTGAN", "BayesianNet", "PrivBayes",], 
                "\nEnter generator to evaluate:\n", 
                numbered=True,
                )
            print("\nWaiting for agent response...\n")
            continue
        
        else: # discuss
            feedback = pyip.inputStr("\nWrite feedback or questions about the generator choice\n")
            feedback_hist.append(feedback)
            print("\nWaiting for agent response...\n")
            continue
        
    return synthesis_request
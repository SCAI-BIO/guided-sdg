from pathlib import Path
import subprocess as sp
from typing import Literal, Any

import pandas as pd
import pyinputplus as pyip
from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agentic_synth_data.prompts.generator_selection import generator_selection_message
from agentic_synth_data.utils import format_response

load_dotenv()


class SelectGenerator(BaseModel):
    generator: Literal["PrivBayes", "CTGAN", "PATEGAN", "BayesianNet"]
    pros: str
    cons: str
    rationale: str
    warning: list[str]


# TODO: If run_generator fails, for example, plug the error message into the agent and try to
# have the agent repair the issue.
def run_generator(data_path:str, clean_csv_path:str, synthesis_request:dict[str, Any],) -> str:
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
        synthesis_request (dict[str, Any]): Dictionary of the user's preferences, with the 
                        decided generator.

    Returns:
        str: Path of the synthetic dataset.                                  
    """
    mechanism = synthesis_request["generator"]
    # Set sample size
    if synthesis_request["sample_size"]["mode"] == "fixed value":
        sample_size = synthesis_request["sample_size"]["fixed"]["value"]
    else:
        sample_size = pd.read_csv(data_path).shape[0]

    # Set cleaned data and output paths
    data_name = Path(clean_csv_path).stem
    output_filename = f"synthetic_{data_name}_{mechanism.lower()}.csv"
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


# TODO: Expand to parameter selection.
def select_generator(
        synthesis_request:dict[str, Any], 
        semantic_types:dict[str, str], 
        agent:CompiledStateGraph, 
        config:dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
    """Based on the user's preferences and information about the dataset, the agent 
    selects a data generator mechanism.

    Args:
        synthesis_request (dict[str, Any]): Dictionary of the user's preferences.
        semantic_types (dict[str, str]): The semantic dtypes for each column in the dataset.
        agent (CompiledStateGraph): LangGraph agent used to recommend a mechanism/generator.
        config (dict[str, dict[str, Any]]: Configuration passed to the agent during invocation.

    Returns:
        dict[str, Any]: Synthesis request with updated generator information.                       
    """
    feedback_hist = []
    selected_mechanism = None
    choice_confirmed = False

    while not choice_confirmed:
        message = generator_selection_message(synthesis_request=synthesis_request, semantic_types=semantic_types,)

        if selected_mechanism is not None:
            message += f"""
The following generator is being considered:
{selected_mechanism}

Give pros, cons, rationale, and warnings for this selected generator.
Do not automatically change the generator unless there is a strong reason.
"""
        
        if feedback_hist:
            message += "\n\nUser feedback so far:\n" + "\n".join(feedback_hist)

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
            },
            config=config
        )

        plan = response['messages'][-1].content.split("</think>")[1].replace("\n", "")
        generation_plan = SelectGenerator.model_validate_json(plan)

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
            synthesis_request["generator"] = generation_plan.generator
            choice_confirmed = True

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
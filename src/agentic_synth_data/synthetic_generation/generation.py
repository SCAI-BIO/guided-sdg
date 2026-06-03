import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal, Any
import pyinputplus as pyip
from pathlib import Path
import subprocess as sp

from langgraph.graph.state import CompiledStateGraph

from agentic_synth_data.prompts.generator_selection import generator_selection_message
from agentic_synth_data.utils import format_response

load_dotenv()


class SelectGenerator(BaseModel):
    generator: Literal["PrivBayes", "CTGAN", "PATEGAN", "BayesianNet"]
    pros: str
    cons: str
    rationale: str
    warning: list[str]


# TODO: add docstring.
def run_generator(data_path:str, clean_csv_path:str, synthesis_request:dict,) -> str:
    # TODO: fix the number of samples, if specified only generating the number of columns present in the df.
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


# TODO: Update docstring
def select_generator(synthesis_request:dict[str, Any], semantic_types:dict[str, str], agent:CompiledStateGraph, config:dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Based on the user's preferences and information about the dataset, the agent selects a data generator mechanism.

    Args:
        synthesis_request (pd.DataFrame): 
                           missing values.
        synthesis_request (pd.DataFrame): 
                           missing values.
        synthesis_request (pd.DataFrame): 
                synthesis_request (pd.DataFrame):

    Returns:
        tuple[dict[str, Any], str | None]:
        - (dict[str, Any]) The synthesis_request (either updated or not).
        - (str | None) The path of the synthetic data.                                    
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
        print(f"Data generator: {generation_plan.generator}")#['generator']}")

        print("\nPros:")
        format_response(generation_plan.pros)#['pros'])

        print("\nCons:")
        format_response(generation_plan.cons)#['cons'])

        print("\nRationale:")
        format_response(generation_plan.rationale)#['rationale'])

        if len(generation_plan.warning) > 0:#["warning"]) > 0:
            print("\nWarning:")
            format_response(generation_plan.warning[0])#['warning'][0])

        print("\nDo you approve using the suggested generator?\n")
        decision = pyip.inputMenu(["approve", "choose another", "discuss",], numbered=True,)

        if decision == "approve":
            synthesis_request["generator"] = generation_plan.generator#["generator"]
            choice_confirmed = True

        elif decision == "choose another":
            selected_mechanism = pyip.inputMenu(
                ["PATEGAN", "CTGAN", "BayesianNet", "PrivBayes",], 
                "\nEnter generator to evaluate:\n", 
                numbered=True,
                )
            continue
        
        else: # discuss
            feedback = pyip.inputStr("\nWrite feedback or questions about the generator choice\n")
            feedback_hist.append(feedback)
            continue
        
        return synthesis_request
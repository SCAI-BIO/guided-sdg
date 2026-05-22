import pandas as pd
from dotenv import load_dotenv
from typing import Literal, Any
import pyinputplus as pyip
from pathlib import Path
import subprocess as sp

from langgraph.types import Command
from langchain.tools import tool, BaseTool

from agentic_synth_data.prompts.generator_selection import generator_selection_message

load_dotenv()


def make_generator_tool(self) -> BaseTool:
    """Make the tool for an agent to generate synthetic data that only allows it
    to choose the mechanism.

    Raises:
        RuntimeError: If the specified generator is unable to generate synthetic data.

    Returns:
        BaseTool: The tool the agent can access (limited to model choice).
    """
    @tool("run_synthetic_generator", description="Run the selected synthetic data generator.",) #return_direct=False # Have the tool immediately direct to user (default)
    def run_synthetic_generator(mechanism:Literal["PATEGAN", "CTGAN", "BayesianNet", "PrivBayes",]) -> str:
        # Set sample size
        if self.synthesis_request["sample_size"]["mode"] == "fixed_value":
            sample_size = self.synthesis_request["sample_size"]["fixed"]["value"]
        else:
            sample_size = pd.read_csv(self.data_path).shape[0]

        # Set cleaned data and output paths
        data_name = Path(self.clean_csv_path).stem
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

        print(f"\n{mechanism}: {sample_size} synthetic samples successfully generated.")
        return str(host_output_path)
    
    return run_synthetic_generator


# TODO: Update docstring (more on choices etc.)
def run_synthesis(self) -> tuple[dict[str, Any], str | None]:
    """Generate the synthetic data.

    Human In The Loop is incorporated here with the choices:
    - approve
    - edit
    - respond
    - reject

    Returns:
        tuple[dict[str, Any], str | None]:
        - (dict[str, Any]) The synthesis_request (either updated or not).
        - (str | None) The path of the synthetic data.                                    
    """
    # TODO: Add a case where the model is already selected by the user and 
        # give a chance about agent feedback (pros/cons) regarding the generator selected
        # then, this would give the option for the user to still update how it is generated
    message = generator_selection_message(
        synthesis_request=self.synthesis_request,
        semantic_types=self.semantic_types,
        clean_csv_path=self.clean_csv_path, # Still not the best to put the path through the agent
        )
    
    input_to_stream = {
        "messages": [
            {
                "role": "user",
                "content": message,
            }
        ]
    }
    self.synthetic_path = None
    while True:
        interrupted = False
        message_buffer = ""

        for chunk in self.agent.stream(
            input_to_stream, 
            config=self.config, 
            stream_mode=["updates", "messages"], 
            version="v2",
            ):
            if chunk["type"] == "messages":
                msg = chunk["data"][0]
                content = getattr(msg, "content", "")
                
                if content:
                    message_buffer += content

                # Capture tool return value
                if msg.__class__.__name__ == "ToolMessage":
                    self.synthetic_path = msg.content

            if chunk["type"] == "updates" and "__interrupt__" in chunk["data"]:
                interrupted = True

                if message_buffer.strip():
                    print(message_buffer.split("</think>")[1].replace("\n", ""))

                print("\nINTERRUPTED:")
                # print(chunk["data"]["__interrupt__"])
                
                # To extract the suggested mechanism (if approved, added to synthesis_request)
                # made single
                action_requests = chunk["data"]["__interrupt__"][0].value["action_requests"][0]

                break

        if not interrupted:
            break    

        decision = pyip.inputMenu(["approve", "edit", "respond", "reject"], numbered=True,)

        # Might help to build response nodes (e.g., approval, response, etc with no need for redefinition if used in later parts)
        if decision == "approve":
            input_to_stream = Command(
                resume={
                    "decisions": [
                        {"type": "approve"}
                    ]
                }
            )
            self.synthesis_request["generator"] = action_requests["args"]["mechanism"]

        elif decision == "edit":
            new_mechanism = pyip.inputMenu(["PATEGAN", "CTGAN", "BayesianNet", "PrivBayes",], "\nEnter replacement generator:\n", numbered=True,)

            input_to_stream = Command(
                resume={
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": self.generator_tool.name,
                                "args": {
                                    "mechanism": new_mechanism
                                },
                            },
                        }
                    ]
                }
            )
            # TODO: Add that if "edit" is selected, the user still can get the model's input on the decision.
            self.synthesis_request["generator"] = new_mechanism

        elif decision == "respond":
            # TODO: Fix required. If user adds a response, no model is suggested and 
            # the process stops running. Check if I forgot to connect something.
            user_message = pyip.inputStr("\nWrite feedback regarding the data and tool choice:\n")
            input_to_stream = Command(
                resume={
                    "decisions": [
                        {
                            "type": "respond",
                            "message": user_message,
                        }
                    ]
                }
            )

        else:
            # In the basic human in the loop, but respond is likely sufficient
            input_to_stream = Command(
                resume={
                    "decisions": [
                        {
                            "type": "reject",
                            "message": "User rejected generator execution."
                        }
                    ]
                }
            )
    return self.synthesis_request, self.synthetic_path
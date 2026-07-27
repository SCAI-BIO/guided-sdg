import textwrap
from typing import Literal, Optional, Type, TypeVar

from pydantic import BaseModel
from langgraph.graph.state import CompiledStateGraph

from agentic_synth_data.prompts.system_prompt import json_repair_prompt

T = TypeVar("T", bound=BaseModel)

def format_response(text:str):
    print(textwrap.fill(text, width=100))


def display_synthesis_request(synthesis_request:"SynthesisRequest") -> None:
    print(f"\nSample size mode: {synthesis_request.sample_size.mode}")

    if synthesis_request.sample_size.fixed_value is not None:
        print(f"Sample size: {synthesis_request.sample_size.fixed_value}")

    print(f"Privacy mode: {synthesis_request.privacy.mode}")

    print(f"Generator: {synthesis_request.generator.name}") 

    print(f"Missing value handling: {synthesis_request.missing_values.mode}")

    if synthesis_request.missing_values.mode == "impute":
        print(f"- Imputer: {synthesis_request.missing_values.impute.method}")
        print(f"- Categorical strategy: {synthesis_request.missing_values.impute.categorical_strategy}")
        print(f"- Numerical strategy: {synthesis_request.missing_values.impute.numerical_strategy}")

    elif synthesis_request.missing_values.mode == "threshold":
        print(f"- Threshold: {synthesis_request.missing_values.threshold}")


def parse_json(
        plan:str, model_cls:Type[BaseModel], agent:Optional[CompiledStateGraph], config:dict, root:bool=False
        ) -> T:
    """If agent fails to correctly format JSON response, prompts to 
    try again.

    Args:
        plan (str): The agent output.
        model_cls (Type[BaseModel]): Pydantic model class used to validate JSON response.
        agent (Optional[CompiledStateGraph]): LangGraph agent.
        config (dict): Configuration passed to the agent during invocation.
        root (bool, optional): Indicates whether in semantic_type_handling (True) or not. Defaults to False.

    Returns:
        T: Validated instance of the given Pydantic model.
    """
    if root: # semantic_type_handling follows a different structure than the other steps
        try:
            return model_cls.model_validate_json(plan).root

        except Exception as error:
            print("\nAgent response could not be parsed. Asking agent to repair the response...")

            repair_prompt = json_repair_prompt(error, plan, model_cls)

            repair_raw = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": repair_prompt,
                        }
                    ]
                },
                config=config,
            )

            repaired_plan = extract_output(repair_raw)

            return model_cls.model_validate_json(repaired_plan).root
    else:
        try:
            return model_cls.model_validate_json(plan)

        except Exception as error:
            print("\nAgent response could not be parsed. Asking agent to repair the response...")

            repair_prompt = json_repair_prompt(error, plan, model_cls)

            repair_raw = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": repair_prompt,
                        }
                    ]
                },
                config=config,
            )

            repaired_plan = extract_output(repair_raw)

            return model_cls.model_validate_json(repaired_plan)


def extract_output(raw_message:dict) -> str:
    """Extract content from the agent's invocation output

    Args:
        raw_message (dict): Raw agent invocation output.

    Returns:
        str: Agent message.
    """
    response = raw_message["messages"][-1].content
    
    if "</think>" in response:
        return response.split("</think>")[1].strip()
    
    return response.strip()


def newline_every_n_words(option:str, n:int) -> str:
    """Format the options output by the agent to improve the readability of pyinputplus's 
    choice menu.

    Args:
        option (str): The option output by the agent.
        n (int): The number of words allowed before a newline is added.

    Returns:
        str: The option output by agent with a newline every n words.
    """
    words = option.split()
    lines = []

    for i in range(0, len(words), n):
        lines.append(" ".join(words[i:i+n]))

    return "\n".join(lines)


def output_default_parameters(
        generator_name:Literal["PATEGAN", "BayesianNet", "PrivBayes", "CTGAN", "auto"]
    ) -> str | None:
    """Takes in generator name and outputs its default parameters. Used in select preferences
    and other instances where parameters are set to default.

    Args:
        generator_name (Literal["PATEGAN", "BayesianNet", "PrivBayes", "CTGAN", "auto"]): _description_

    Returns:
        str | None: 
        - str: The colon-separated generator parameters.
        - None: If auto is selected, no parameters are returned.
    """
    if generator_name == "PATEGAN":
       return "1,1e-5,False,10,100,128,1e-4,False"

    elif generator_name == "BayesianNet":
        return "10,1,False,True,None"

    elif generator_name == "PrivBayes":
        return "10,1,.1,False,True,None"

    elif generator_name == "CTGAN":
        return "128,(256, 256),(256, 256),1e-6,500,300,False"

    else: # "auto"
        return None

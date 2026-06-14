import textwrap
from typing import Optional, Type

from pydantic import BaseModel
from langgraph.graph.state import CompiledStateGraph

from agentic_synth_data.prompts.system_prompt import json_repair_prompt


def format_response(text:str):
    print(textwrap.fill(text, width=100))


def parse_json(
        plan:str, model_cls:Type[BaseModel], agent:Optional[CompiledStateGraph], config:dict, root:bool=False
        ) -> BaseModel:
    """If agent fails to correctly format JSON response, prompts to 
    try again.

    Args:
        plan (str): The agent output.
        model_cls (Type[BaseModel]): Pydantic model class used to validate JSON response.
        agent (Optional[CompiledStateGraph]): LangGraph agent.
        config (dict): Configuration passed to the agent during invocation.
        root (bool, optional): Indicates whether in semantic_type_handling (True) or not. Defaults to False.

    Returns:
        BaseModel: Validated instance of the given Pydantic model.
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

            repair_response = repair_raw["messages"][-1].content
            repaired_plan = extract_output(repair_response)

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

            repair_response = repair_raw["messages"][-1].content
            repaired_plan = extract_output(repair_response)

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
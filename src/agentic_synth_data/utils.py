import textwrap
from typing import Optional

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel


def format_response(text:str):
    print(textwrap.fill(text, width=100))

def json_repair_prompt(error, plan, model_cls):
    return f"""
The previous response could not be parsed as valid JSON.

Error:
{error}

Invalid response:
{plan}

Return exactly one valid JSON object matching this schema:
{model_cls.model_json_schema()}

No markdown.
No explanation outside JSON.
"""

# TODO update typing and docstring
def parse_json(plan, model_cls, agent:Optional[CompiledStateGraph], config:dict, root:bool=False):
    """If agent fails to correctly format JSON response, prompts to 
    try again."""
    if root: # semantic type handling follows a diffrent structure from other steps
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
    

# TODO update typing and docstring
def extract_output(raw_message):
    """Extract content from the agent's invocation output"""
    
    response = raw_message["messages"][-1].content
    
    if "</think>" in response:
        return response.split("</think>")[1].strip()
    
    return response.strip()
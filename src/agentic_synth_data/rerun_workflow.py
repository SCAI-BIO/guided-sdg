from typing import Any, Literal, Optional

from langgraph.graph.state import CompiledStateGraph
import pyinputplus as pyip
from pydantic import BaseModel, Field

from agentic_synth_data.preference_setup import SynthesisRequest
from agentic_synth_data.prompts.revision_prompt import revision_message
from agentic_synth_data.utils import parse_json, extract_output


class RevisionOption(BaseModel):
    id: str
    rerun_step: Literal["generator_selection"]
    generator_name: Literal["CTGAN", "PATEGAN", "BayesianNet", "PrivBayes"]
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str


class RevisionPlan(BaseModel):
    revision_options: list[RevisionOption]


def plan_update(
        synthesis_request:SynthesisRequest, 
        agent_decision:dict[str, Any], 
        agent:Optional[CompiledStateGraph], 
        config:dict,
        ) -> RevisionOption | None:
    """Agent suggests possible retroactive changes, which the user can select from.
    This is limited to generator selection only.
    Returns revision plan"""
    # TODO: remind the agent about the dataset's information (possibly set up mroe like generator selection?)
    # TODO: Go over each step section (mainly generator selection and missing value handling)
    # have the agent output steps it would consider changing (in this case, (for now, leave it only at only changing the generator selection/parameters))
    prompt = revision_message(synthesis_request, agent_decision)
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
    revision_plan = parse_json(plan, RevisionPlan, agent, config)

    menu_options = []
    for option in revision_plan.revision_options:
        label = f"{option.id}: {option.generator_name} - {option.reason}"
        menu_options.append(label)

    choice = pyip.inputMenu(menu_options + ["Cancel"], "\nSelect a revision option:\n", numbered=True)
    # TODO: Incorporate multiple comma selection

    if choice == "Cancel":
        print("\n=> No revision has been selected, the workflow has finished.")
        return None
    
    selected_id = choice.split(":")[0]

    for option in revision_plan.revision_options:
        if option.id == selected_id:
            return option  
    return None
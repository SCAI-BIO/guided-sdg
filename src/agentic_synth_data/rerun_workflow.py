from typing import Any, Literal, Optional

from langgraph.graph.state import CompiledStateGraph
import pyinputplus as pyip
from pydantic import BaseModel, Field

from agentic_synth_data.report import Report
from agentic_synth_data.schemas.models import AgentEval
from agentic_synth_data.preference_setup import SynthesisRequest
from agentic_synth_data.prompts.revision_prompt import revision_message, missing_revision_message
from agentic_synth_data.utils import parse_json, extract_output, newline_every_n_words


class GeneratorRevisionOption(BaseModel):
    id: str
    generator_name: Literal["CTGAN", "PATEGAN", "BayesianNet", "PrivBayes"]
    parameters: dict[str, Any] = Field(default_factory=dict) # NOTE: Not certain whether to implement together (user selection).
    reason: str


class GeneratorRevisionPlan(BaseModel):
    options: list[GeneratorRevisionOption]

# TODO: add parameter revision (make user selectable?)

class MissingRevisionOption(BaseModel):
    id: str
    mode: Literal["remove", "impute",] # Did not include threshold, it isn't safe (some nan might be present)
    #set_threshold: int | None = None
    imputation_method: Literal["SimpleImputer"] | None = None
    categorical_strategy: Literal["most_frequent", "constant"] | None = None
    numerical_strategy: Literal["mean", "median", "most_frequent"] | None = None
    reason: str


class MissingRevisionPlan(BaseModel):
    options: list[MissingRevisionOption]


def plan_update(
        synthesis_request:SynthesisRequest, 
        agent_decision:AgentEval, 
        report:Report,
        agent:Optional[CompiledStateGraph], 
        config:dict,
        ) -> dict | None:
    """Agent suggests possible retroactive changes, which the user can select from.
    
    Steps that can be updated:
    - Generator selection (no params yet)
    - Missing handling

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent_decision (dict[str, Any]): Agent decision based on results.
        report (Report): Report containing information gathered throughout the workflow steps.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to update the current data synthesis plan.
        config (dict): Configuration passed to the agent during invocation.

    Returns:
        dict | None: If a revision was chosen, returns the update_plan dict. Otherwise, None.
    """
    # Select mechanism
    update_mech = generator_update(synthesis_request, agent_decision, report, agent, config)

    # Mechanism parameters
    # update_params should have a similar structure (but depends on if user should be selecting the option)

    # Missing handling
    update_missing = None

    if update_mech is not None:
        effective_generator = update_mech.generator_name
    else:
        effective_generator = synthesis_request.generator.name

    if effective_generator in ["PrivBayes", "BayesianNet"]:
        print("\n=> The data generator selected allows for missing values. Missing value handling will not be applied.")
    else: 
        update_missing = missing_update(synthesis_request, agent, config)

    update_plan = {} 
    if update_mech:
        update_plan["update_mech"] = update_mech
    
    # if update_params: # TODO: implement this
    #     updated_plan["update_params"] = update_params

    if update_missing is not None:
        update_plan["update_missing"] = update_missing

    return update_plan if update_plan else None


def generator_update(
        synthesis_request:SynthesisRequest, 
        agent_decision:AgentEval, 
        report:Report,
        agent:Optional[CompiledStateGraph], 
        config:dict,
        ) -> GeneratorRevisionOption | None:
    """Ask agent for for possible generator revision options and returns the option selected by the user.

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent_decision (AgentEval): Agent decision on results: satisfactory | not_satisfactory | needs_review.
        report (Report): Report containing information gathered throughout the workflow steps.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a generator.
        config (dict): Configuration passed to the agent during invocation.

    Returns:
        GeneratorRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    prompt = revision_message(synthesis_request, report, agent_decision)
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
    revision_plan = parse_json(plan, GeneratorRevisionPlan, agent, config)

    menu_options = {}
    for option in revision_plan.options:
        label = newline_every_n_words(f"{option.generator_name} - {option.reason}", 12)
        menu_options[label] = option

    choice = pyip.inputMenu(list(menu_options.keys()) + ["Cancel"], "\nSelect a generator revision option:\n", numbered=True)

    if choice == "Cancel":
        print("\n=> No revision has been selected for generator choice, continuing to missing value handling.")
        return None
    
    return menu_options[choice]


def missing_update(
        synthesis_request:SynthesisRequest, 
        agent:Optional[CompiledStateGraph], 
        config:dict,
        ) -> MissingRevisionOption | None:
    """
    Ask agent for for possible generator revision options and returns the option selected by the user.

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent (Optional[CompiledStateGraph]):  LangGraph agent used to recommend a strategy for missing values.
        config (dict): Configuration passed to the agent during invocation.

    Returns:
        MissingRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    prompt = missing_revision_message(synthesis_request)
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
    revision_plan = parse_json(plan, MissingRevisionPlan, agent, config)

    menu_options = {}
    for option in revision_plan.options:
        if option.mode == "remove":
            label = newline_every_n_words(f"Mode: {option.mode} - {option.reason}", 12)

        if option.mode == "impute":
            label = f"Mode: {option.mode}, Numerical strategy: {option.numerical_strategy}, Categorical strategy: {option.categorical_strategy}\n- {newline_every_n_words(option.reason, 12)}"
        
        menu_options[label] = option

    choice = pyip.inputMenu(list(menu_options.keys()) + ["Cancel"], "\nSelect a missing value revision option:\n", numbered=True)

    if choice == "Cancel":
        print("\n=> No revision has been selected for missing value handling. Previously selected settings will be used.")
        return None
    
    return menu_options[choice]
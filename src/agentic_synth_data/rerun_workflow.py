from typing import Any, Literal, Optional

from langgraph.graph.state import CompiledStateGraph
import pyinputplus as pyip
from pydantic import BaseModel, Field
import streamlit as st

from agentic_synth_data.report import Report
from agentic_synth_data.schemas.models import AgentEval
from agentic_synth_data.preference_setup import SynthesisRequest
from agentic_synth_data.prompts.revision_prompt import revision_message, missing_revision_message, \
    parameter_revision_message
from agentic_synth_data.utils import parse_json, extract_output, newline_every_n_words


PENDING = "pending"

class GeneratorRevisionOption(BaseModel):
    id: str
    generator_name: Literal["CTGAN", "PATEGAN", "BayesianNet", "PrivBayes"]
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str


class GeneratorRevisionPlan(BaseModel):
    options: list[GeneratorRevisionOption]


class PATEGANConfig(BaseModel):
    eps: int = Field(default=1, ge=0, le=7)


class PrivBayesConfig(BaseModel):
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)


CONFIG_MAP = {
   "PATEGAN": PATEGANConfig,
   "PrivBayes": PrivBayesConfig,
}


class MissingRevisionOption(BaseModel):
    id: str
    mode: Literal["remove", "impute",]
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
        agent (Optional[CompiledStateGraph]): LangGraph agent used to update the current data
                        synthesis plan.
        config (dict): Configuration passed to the agent during invocation.

    Returns:
        dict | None: If a revision was chosen, returns the update_plan dict. Otherwise, None.
    """
    # Select mechanism
    update_mech = generator_update(synthesis_request, agent_decision, report, agent, config)

    if update_mech is not None:
        effective_generator = update_mech.generator_name
    else:
        effective_generator = synthesis_request.generator.name

    # Mechanism parameters
    update_params = parameter_update(effective_generator, agent, config)

    # Missing handling
    update_missing = None

    if effective_generator in ["PrivBayes", "BayesianNet"]:
        print(
            "\n=> The data generator selected allows for missing values. Missing value handling" \
            " will not be applied."
        )
    else:
        update_missing = missing_update(synthesis_request, agent, config)

    update_plan = {}
    if update_mech:
        update_plan["update_mech"] = update_mech

    if update_params:
        update_plan["update_params"] = update_params

    if update_missing is not None:
        update_plan["update_missing"] = update_missing

    return update_plan if update_plan else None


def plan_update_ui(
    synthesis_request:SynthesisRequest,
    agent_decision:AgentEval,
    report:Report,
    agent:Optional[CompiledStateGraph],
    config:dict,
    version_id:int,
) -> dict | None:
    """Agent suggests possible retroactive changes, which the user can select from (UI).
    
    Steps that can be updated:
    - Generator selection (no params yet)
    - Missing handling

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent_decision (dict[str, Any]): Agent decision based on results.
        report (Report): Report containing information gathered throughout the workflow steps.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to update the current data
                        synthesis plan.
        config (dict): Configuration passed to the agent during invocation.
        version_id (int): Denotes the version of the workflow attempt.

    Returns:
        dict | None: If a revision was chosen, returns the update_plan dict. Otherwise, None.
    """
    # Update generator
    update_mech = generator_update_ui(
        synthesis_request, agent_decision, report, agent, config, version_id,
    )

    if update_mech == PENDING:
        st.stop()

    if update_mech is not None:
        effective_generator = update_mech.generator_name
    else:
        effective_generator = synthesis_request.generator.name

    # Update generator parameters
    update_params = parameter_update_ui(effective_generator, agent, config,)

    if update_params == PENDING:
        st.stop()

    # Update missing handling
    update_missing = None

    if effective_generator in ["PrivBayes", "BayesianNet"]:
        st.info(
            "=> The data generator selected allows for missing values. Missing value handling"
            " will not be applied."
        )
    else:
        update_missing = missing_update_ui(synthesis_request, agent, config, version_id,)

        if update_missing == PENDING:
            st.stop()

    # Setup update plan
    update_plan = {}
    if update_mech:
        update_plan["update_mech"] = update_mech

    if update_params:
        update_plan["update_params"] = update_params

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
    """Ask agent for for possible generator revision options and returns the option selected by 
    the user.

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent_decision (AgentEval): Agent decision on results: satisfactory | not_satisfactory |
                                    needs_review.
        report (Report): Report containing information gathered throughout the workflow steps.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a generator.
        config (dict): Configuration passed to the agent during invocation.

    Returns:
        GeneratorRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    print("\nAgent is coming up with a generator revision options...")
    prompt = revision_message(synthesis_request, report, agent_decision)
    raw_response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt,}]},
        config=config,
    )

    plan = extract_output(raw_response)
    revision_plan = parse_json(plan, GeneratorRevisionPlan, agent, config)

    menu_options = {}
    for option in revision_plan.options:
        label = newline_every_n_words(f"{option.generator_name} - {option.reason}", 12)
        menu_options[label] = option

    choice = pyip.inputMenu(
        list(menu_options.keys()) + ["Cancel"],
        "\nSelect a generator revision option:\n", 
        numbered=True
    )

    if choice == "Cancel":
        print(
            "\n=> No revision has been selected for generator choice, continuing to " \
            "missing value handling."
        )
        return None

    return menu_options[choice]


def generator_update_ui(
    synthesis_request:SynthesisRequest,
    agent_decision:AgentEval,
    report:Report,
    agent:Optional[CompiledStateGraph],
    config:dict,
    version_id:int,
) -> GeneratorRevisionOption | None | Literal["pending"]:
    """Ask agent for for possible generator revision options and returns the option selected by 
    the user (UI).

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent_decision (AgentEval): Agent decision on results: satisfactory | not_satisfactory |
                                    needs_review.
        report (Report): Report containing information gathered throughout the workflow steps.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a generator.
        config (dict): Configuration passed to the agent during invocation.
        version_id (int): Denotes the version of the workflow attempt.

    Returns:
        GeneratorRevisionOption | None | Literal["pending"]: Selected generator revision, or None 
                                if cancelled. Literal["pending"] if nothing has been selected yet
                                from the selectbox.
    """
    st.subheader("Select a generator revision option")
    if f"generator_revision_plan_{version_id}" not in st.session_state:
        with st.spinner("Agent is coming up with a generator revision options..."):
            prompt = revision_message(synthesis_request, report, agent_decision)
            raw_response = agent.invoke(
                {"messages": [{"role": "user", "content": prompt,}]},
                config=config,
            )

            plan = extract_output(raw_response)
            revision_plan = parse_json(plan, GeneratorRevisionPlan, agent, config)

        st.session_state.generator_revision_plan = revision_plan

    else:
        revision_plan = st.session_state.generator_revision_plan

    # TODO: add an info box that discusses what was input into the revision message

    menu_options = {}

    for option in revision_plan.options:
        menu_options[option.generator_name] = option
        st.write(f"- {option.generator_name}: {option.reason}")

    choice = st.selectbox(
        "Select a generator revision option",
        list(menu_options.keys()) + ["Cancel"],
        index=None,
        key=f"rerun_generator_update_{version_id}",
    )

    if choice is None:
        return PENDING

    if choice == "Cancel": # TODO maybe remove cancel for ui?
        st.info(
            "=> No revision has been selected for generator choice, continuing to "
            "missing value handling."
        )
        return None

    return menu_options[choice]


def parameter_update(
        selected_mechanism:Literal["PATEGAN", "CTGAN", "PrivBayes", "BayesianNet"],
        agent:Optional[CompiledStateGraph],
        config:dict,
    ) -> str | None:
    """Ask agent for possible parameter revision.

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a generator.
        config (dict): Configuration passed to the agent during invocation.

    Returns:
        GeneratorRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    if selected_mechanism in ["BayesianNet", "CTGAN"]:
        print(
            f"=> {selected_mechanism} can only be set to default." \
              " Default parameters retained."
        )
        return None

    print("\nAgent is coming up with a parameter revision...")
    prompt = parameter_revision_message(selected_mechanism)
    raw_response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt,}]},
        config=config,
    )

    raw_params = extract_output(raw_response)

    param_revision = parse_json(raw_params, CONFIG_MAP[selected_mechanism], agent, config)

    changed_params = param_revision.model_dump(exclude_defaults=True)

    if not changed_params:
        print("\n=> Default parameters retained.")
        return None

    if selected_mechanism == "PATEGAN":
        mech_params = f"{param_revision.eps},1e-5,False,10,100,128,1e-4,False"
        selection = ("epsilon", 1, param_revision.eps)

    elif selected_mechanism == "PrivBayes":
        mech_params = f"10,1,{param_revision.epsilon},False,True,None"
        selection = ("epsilon", .1, param_revision.epsilon)

    print(
        f"\n=> {selected_mechanism}: {selection[0]} has been updated to {selection[2]} " \
        f"(default = {selection[1]})"
    )
    return mech_params


def parameter_update_ui(
    selected_mechanism:Literal["PATEGAN", "CTGAN", "PrivBayes", "BayesianNet"],
    agent:Optional[CompiledStateGraph],
    config:dict,
) -> str | None | Literal["pending"]:
    """Ask agent for possible parameter revision (UI).

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a generator.
        config (dict): Configuration passed to the agent during invocation.

    Returns:
        GeneratorRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    if selected_mechanism in ["BayesianNet", "CTGAN"]:
        st.info(
            f"=> {selected_mechanism} can only be set to default."
            " Default parameters retained."
        )
        return None

 # TODO add a check for the agent to only suggest once
    with st.spinner("Agent is coming up with a parameter revision..."):
        prompt = parameter_revision_message(selected_mechanism)
        raw_response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt,}]},
            config=config,
        )

        raw_params = extract_output(raw_response)

        param_revision = parse_json(raw_params, CONFIG_MAP[selected_mechanism], agent, config)

    changed_params = param_revision.model_dump(exclude_defaults=True) # TODO? move to indent?

    if not changed_params:
        st.info("\n=> Default parameters retained.")
        return None

    if selected_mechanism == "PATEGAN":
        mech_params = f"{param_revision.eps},1e-5,False,10,100,128,1e-4,False"
        selection = ("epsilon", 1, param_revision.eps)

    elif selected_mechanism == "PrivBayes":
        mech_params = f"10,1,{param_revision.epsilon},False,True,None"
        selection = ("epsilon", .1, param_revision.epsilon)

    st.info(
        f"\n=> {selected_mechanism}: {selection[0]} has been updated to {selection[2]} "
        f"(default = {selection[1]})"
    )
    return mech_params


def missing_update(
        synthesis_request:SynthesisRequest,
        agent:Optional[CompiledStateGraph],
        config:dict,
    ) -> MissingRevisionOption | None:
    """
    Ask agent for possible generator revision options and returns the option selected by the 
    user.

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a strategy for 
                                              missing values.
        config (dict): Configuration passed to the agent during invocation.

    Returns:
        MissingRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    print("\nAgent is coming up with missing value handling options...")
    prompt = missing_revision_message(synthesis_request)
    raw_response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt,}]},
        config=config,
    )

    plan = extract_output(raw_response)
    revision_plan = parse_json(plan, MissingRevisionPlan, agent, config)

    menu_options = {}
    for option in revision_plan.options:
        if option.mode == "remove":
            label = newline_every_n_words(f"Mode: {option.mode} - {option.reason}", 12)

        if option.mode == "impute":
            label = f"Mode: {option.mode}, Numerical strategy: {option.numerical_strategy}, " \
            f"Categorical strategy: {option.categorical_strategy}\n- " \
            f"{newline_every_n_words(option.reason, 12)}"

        menu_options[label] = option

    choice = pyip.inputMenu(
        list(menu_options.keys()) + ["Cancel"],
        "\nSelect a missing value revision option:\n", 
        numbered=True
        )

    if choice == "Cancel":
        print(
            "\n=> No revision has been selected for missing value handling. " \
            "Previously selected settings will be used."
        )
        return None

    return menu_options[choice]


def missing_update_ui(
    synthesis_request:SynthesisRequest,
    agent:Optional[CompiledStateGraph],
    config:dict,
    version_id:int,
) -> MissingRevisionOption | None | Literal["pending"]:
    """
    Ask agent for possible generator revision options and returns the option selected by the 
    user .

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a strategy for 
                                              missing values.
        config (dict): Configuration passed to the agent during invocation.
        version_id (int): Denotes the version of the workflow attempt.

    Returns:
        MissingRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    st.subheader("Select a missing value revision option")

    if f"missing_revision_plan_{version_id}" not in st.session_state:
        with st.spinner("Agent is coming up with missing value handling options..."):
            prompt = missing_revision_message(synthesis_request)
            raw_response = agent.invoke(
                {"messages": [{"role": "user", "content": prompt,}]},
                config=config,
            )

            plan = extract_output(raw_response)
            revision_plan = parse_json(plan, MissingRevisionPlan, agent, config,)

        st.session_state.missing_revision_plan = revision_plan

    else:
        revision_plan = st.session_state.missing_revision_plan

    menu_options = {}

    for option in revision_plan.options:
        if option.mode == "remove":
            st.write(f"- {option.mode}: {option.reason}")
            label = newline_every_n_words(f"Mode: {option.mode} - {option.reason}", 12)

        if option.mode == "impute":
            st.write(f"- {option.mode}: {option.reason}")
            label = f"Mode: {option.mode}, Numerical strategy: {option.numerical_strategy}, " \
            f"Categorical strategy: {option.categorical_strategy}\n- " \
            f"{newline_every_n_words(option.reason, 12)}"

        menu_options[label] = option

    st.write("Select a missing value revision option:")
    choice = st.selectbox(
        "selecting a revision option",
        list(menu_options.keys()) + ["Cancel"],
        index=None,
        label_visibility="hidden",
    )

    if choice is None:
        return PENDING

    if choice == "Cancel":
        st.info(
            "=> No revision has been selected for missing value handling. "
            "Previously selected settings will be used."
        )
        return None

    return menu_options[choice]

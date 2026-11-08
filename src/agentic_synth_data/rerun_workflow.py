from typing import Any, Literal, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
import pyinputplus as pyip
import streamlit as st

from agentic_synth_data.report import Report
from agentic_synth_data.schemas.eval_model import AgentEval
from agentic_synth_data.schemas.generator_model import GeneratorRevisionOption, \
    GeneratorRevisionPlan, PATEGANConfig, PrivBayesConfig
from agentic_synth_data.schemas.missing_model import MissingRevisionOption, MissingRevisionPlan
from agentic_synth_data.preference_setup import SynthesisRequest
from agentic_synth_data.prompts.revision_prompt import revision_message, missing_revision_message, \
    parameter_revision_message
from agentic_synth_data.utils import GeneratorName, format_option, info_helper_message, invoke_agent, \
    newline_every_n_words


PENDING = "pending"

CONFIG_MAP = {
   "PATEGAN": PATEGANConfig,
   "PrivBayes": PrivBayesConfig,
}


def plan_update(
    synthesis_request: SynthesisRequest,
    agent_decision: AgentEval,
    report: Report,
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
) -> dict | None:
    """Agent suggests possible retroactive changes, which the user can select from.
    
    Steps that can be updated:
    - Generator selection (no params yet)
    - Missing handling

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent_decision (AgentEval): Agent decision based on results.
        report (Report): Report containing information gathered throughout the workflow steps.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to update the current data
                        synthesis plan.
        config (RunnableConfig): Configuration passed to the agent during invocation.

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
            "\n🠊 The data generator selected allows for missing values. Missing value handling" \
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
    synthesis_request: SynthesisRequest,
    agent_decision: AgentEval,
    report: Report,
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
    version_id: int,
) -> dict | None:
    """Agent suggests possible retroactive changes, which the user can select from (UI).
    
    Steps that can be updated:
    - Generator selection (no params yet)
    - Missing handling

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent_decision (AgentEval): Agent decision based on results.
        report (Report): Report containing information gathered throughout the workflow steps.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to update the current data
            synthesis plan.
        config (RunnableConfig): Configuration passed to the agent during invocation.
        version_id (int): Denotes the version of the workflow attempt.

    Returns:
        dict | None: If a revision was chosen, returns the update_plan dict. Otherwise, None.
    """
    # Update generator
    update_mech = generator_update_ui(
        synthesis_request=synthesis_request,
        agent_decision=agent_decision,
        report=report,
        agent=agent,
        config=config,
        version_id=version_id,
    )

    if update_mech == PENDING:
        st.stop()

    if update_mech is not None:
        effective_generator = update_mech.generator_name
    else:
        effective_generator = synthesis_request.generator.name

    # Update generator parameters
    update_params = parameter_update_ui(
        selected_mechanism=effective_generator,
        agent=agent,
        config=config,
        version_id=version_id,
    )

    if update_params == PENDING:
        st.stop()

    # Update missing handling
    update_missing = None

    if effective_generator in ["PrivBayes", "BayesianNet"]:
        st.info(
            "⇨ The data generator selected allows for missing values. Missing value handling"
            " will not be applied."
        )
    else:
        update_missing = missing_update_ui(
            synthesis_request=synthesis_request,
            agent=agent,
            config=config,
            version_id=version_id,
        )

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
    synthesis_request: SynthesisRequest,
    agent_decision: AgentEval,
    report: Report,
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
) -> GeneratorRevisionOption | None:
    """Ask agent for for possible generator revision options and returns the option selected by 
    the user.

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent_decision (AgentEval): Agent decision on results: satisfactory | not_satisfactory |
                                    needs_review.
        report (Report): Report containing information gathered throughout the workflow steps.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a generator.
        config (RunnableConfig): Configuration passed to the agent during invocation.

    Returns:
        GeneratorRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    print(f"Current generator: {synthesis_request.generator.name}")
    print("\nAgent is coming up with a generator revision options...")

    revision_plan = invoke_agent(
        prompt=revision_message(synthesis_request, report, agent_decision),
        model_cls=GeneratorRevisionPlan,
        agent=agent,
        config=config,
    )

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
            "\n🠊 No revision has been selected for generator choice, continuing to " \
            "missing value handling."
        )
        return None

    return menu_options[choice]


def generator_update_ui(
    synthesis_request: SynthesisRequest,
    agent_decision: AgentEval,
    report: Report,
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
    version_id: int,
) -> GeneratorRevisionOption | None | Literal["pending"]:
    """Ask agent for for possible generator revision options and returns the option selected by 
    the user (UI).

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent_decision (AgentEval): Agent decision on results: satisfactory | not_satisfactory |
            needs_review.
        report (Report): Report containing information gathered throughout the workflow steps.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a generator.
        config (RunnableConfig): Configuration passed to the agent during invocation.
        version_id (int): Denotes the version of the workflow attempt.

    Returns:
        GeneratorRevisionOption | None | Literal["pending"]: Selected generator revision, or None 
            if cancelled. Literal["pending"] if nothing has been selected yet from the selectbox.
    """
    plan_key = f"generator_revision_plan_{version_id}"
    choice_key = f"generator_revision_choice_{version_id}"

    if choice_key in st.session_state:
        return st.session_state[choice_key]

    st.subheader("Select a generator revision option")

    st.write(f"**Current generator:** {synthesis_request.generator.name}")

    if plan_key not in st.session_state:
        with st.spinner("Agent is coming up with a generator revision options..."):
            revision_plan = invoke_agent(
                prompt=revision_message(synthesis_request, report, agent_decision),
                model_cls=GeneratorRevisionPlan,
                agent=agent,
                config=config,
                ui=True,
            )

        st.session_state[plan_key] = revision_plan

    else:
        revision_plan = st.session_state[plan_key]

    info_helper_message(
        message="The agent has finished evaluating generator options.",
        context=(
            "The agent evaluated the synthetic dataset based on:\n"
            "- Your preferences.\n"
            "- The agent's evaluation of the current results.\n"
            "- Information gathered throughout the workflow.\n"
            "   - Synthetic data quality and fidelity metrics.\n"
            "   - Privacy evaluation from Anonymeter.\n"
            "- The current generator configuration."
        ),
        key=f"generator_revision_info_{version_id}",
    )

    st.write("Suggested data generators to switch to:")

    menu_options = {}

    for option in revision_plan.options:
        menu_options[option.generator_name] = option
        st.write(f"- **{option.generator_name}:** {option.reason}")

    choice = st.selectbox(
        label="Select a generator revision option. Select cancel to retain your current selection.",
        options=list(menu_options.keys()) + ["Cancel"],
        index=None,
        key=f"rerun_generator_update_{version_id}",
    )

    if choice is None:
        return PENDING

    # TODO Possibly change the name to keep default (but then should show default selection...)
    if choice == "Cancel":
        st.session_state[choice_key] = None
        st.info(
            "⇨ No revision has been selected for generator choice, continuing to "
            "missing value handling."
        )
        return None

    selected_option = menu_options[choice]
    st.session_state[choice_key] = selected_option

    return selected_option


def parameter_update(
    selected_mechanism: GeneratorName, agent: Optional[CompiledStateGraph], config: RunnableConfig,
) -> str | None:
    """Ask agent for possible parameter revision.

    Args:
        selected_mechanism (GeneratorName): The selected mechanism.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a generator.
        config (RunnableConfig): Configuration passed to the agent during invocation.

    Returns:
        GeneratorRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    if selected_mechanism in ["BayesianNet", "CTGAN"]:
        print(
            f"🠊 {selected_mechanism} can only be set to default." \
              " Default parameters retained."
        )
        return None

    print("\nAgent is coming up with a parameter revision...")

    param_revision = invoke_agent(
        prompt=parameter_revision_message(selected_mechanism),
        model_cls=CONFIG_MAP[selected_mechanism],
        agent=agent,
        config=config,
    )

    changed_params = param_revision.model_dump(exclude_defaults=True)

    if not changed_params:
        print("\n🠊 Default parameters retained.")
        return None

    if selected_mechanism == "PATEGAN":
        mech_params = {
            "eps":param_revision.eps, "delta":1e-5, "infer_ranges":False, "num_teachers":10,
            "n_iters":100, "batch_size":128, "learning_rate":1e-4, "multiprocess":False,
        }
        selection = ("epsilon", 1, param_revision.eps)

    elif selected_mechanism == "PrivBayes":
        mech_params = {
            "histogram_bins":10, "degree":1, "epsilon":param_revision.epsilon, "infer_ranges":False,
            "multiprocess":True, "seed":None,
        }
        selection = ("epsilon", .1, param_revision.epsilon)

    print(
        f"\n🠊 {selected_mechanism}: {selection[0]} has been updated to {selection[2]} " \
        f"(default = {selection[1]})"
    )
    return mech_params


def parameter_update_ui(
    selected_mechanism: GeneratorName,
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
    version_id: int,
) -> dict[str, Any] | None | Literal["pending"]:
    """Ask agent for possible parameter revision (UI).

    Args:
        selected_mechanism (GeneratorName): The selected mechanism.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a generator.
        config (RunnableConfig): Configuration passed to the agent during invocation.
        version_id (int): Denotes the version of the workflow attempt.

    Returns:
        GeneratorRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    if selected_mechanism in ["BayesianNet", "CTGAN"]:
        st.info(
            f"⇨ {selected_mechanism} can only be set to default. Default parameters retained."
        )
        return None

 # TODO add a check for the agent to only suggest once
    with st.spinner("Agent is coming up with a parameter revision..."):
        param_revision = invoke_agent(
            prompt=parameter_revision_message(selected_mechanism),
            model_cls=CONFIG_MAP[selected_mechanism],
            agent=agent,
            config=config,
            ui=True,
        )

    changed_params = param_revision.model_dump(exclude_defaults=True)

    info_helper_message(
        message="The agent has finished determining parameter options.",
        context=(
            "The agent investigated parameter choices based on:\n"
            "- Available parameter options for the selected generator."
        ),
        key=f"parameter_revision_info_{version_id}",
    )

    if not changed_params:
        st.info("⇨ Default parameters retained.")
        return None

    if selected_mechanism == "PATEGAN":
        mech_params = {
            "eps": param_revision.eps, "delta": 1e-5, "infer_ranges": False, "num_teachers": 10,
            "n_iters": 100, "batch_size": 128, "learning_rate": 1e-4, "multiprocess": False,
        }
        selection = ("epsilon", 1, param_revision.eps)

    elif selected_mechanism == "PrivBayes":
        mech_params = {
            "histogram_bins": 10, "degree": 1, "epsilon": param_revision.epsilon,
            "infer_ranges": False, "multiprocess": True, "seed": None,
        }
        selection = ("epsilon", .1, param_revision.epsilon)

    st.info(
        f"⇨ {selected_mechanism}: {selection[0]} has been updated to {selection[2]} "
        f"(default = {selection[1]})"
    )
    return mech_params


def missing_update(
    synthesis_request: SynthesisRequest,
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
) -> MissingRevisionOption | None:
    """
    Ask agent for possible generator revision options and returns the option selected by the 
    user.

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a strategy for 
                                              missing values.
        config (RunnableConfig): Configuration passed to the agent during invocation.

    Returns:
        MissingRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    print(f"Current missing handling method: {synthesis_request.generator.name}")
    print("\nAgent is coming up with missing value handling options...")

    revision_plan = invoke_agent(
        prompt=missing_revision_message(synthesis_request),
        model_cls=MissingRevisionPlan,
        agent=agent,
        config=config,
    )

    menu_options = {}
    for option in revision_plan.options:
        if option.mode == "remove":
            label = newline_every_n_words(f"Mode: {option.mode} - {option.reason}", 12)

        if option.mode == "impute":
            label = f"Mode: {option.mode}, Numerical strategy: {option.numerical_strategy}, " \
            f"Categorical strategy: {format_option(option.categorical_strategy)}\n- " \
            f"{newline_every_n_words(option.reason, 12)}"

        menu_options[label] = option

    choice = pyip.inputMenu(
        list(menu_options.keys()) + ["Cancel"],
        "\nSelect a missing value revision option:\n", 
        numbered=True
        )

    if choice == "Cancel":
        print(
            "\n🠊 No revision has been selected for missing value handling. " \
            "Previously selected settings will be used."
        )
        return None

    return menu_options[choice]


def missing_update_ui(
    synthesis_request: SynthesisRequest,
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
    version_id: int,
) -> MissingRevisionOption | None | Literal["pending"]:
    """
    Ask agent for possible generator revision options and returns the option selected by the 
    user (UI).

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        agent (Optional[CompiledStateGraph]): LangGraph agent used to recommend a strategy for 
            missing values.
        config (RunnableConfig): Configuration passed to the agent during invocation.
        version_id (int): Denotes the version of the workflow attempt.

    Returns:
        MissingRevisionOption | None: Selected generator revision, or None if cancelled.
    """
    st.subheader("Select a missing value revision option")

    plan_key = f"missing_revision_plan_{version_id}"

    if plan_key not in st.session_state:
        with st.spinner("Agent is coming up with missing value handling options..."):
            revision_plan = invoke_agent(
                prompt=missing_revision_message(synthesis_request),
                model_cls=MissingRevisionPlan,
                agent=agent,
                config=config,
                ui=True,
            )

        st.session_state[plan_key] = revision_plan

    else:
        revision_plan = st.session_state[plan_key]

    info_helper_message(
        message="The agent has finished determining parameter options.",
        context=(
            "The agent investigated parameter choices based on:\n"
            "- Your preferences."
        ),
        key=f"missing_revision_info_{version_id}",
    )

    menu_options = {}

    for option in revision_plan.options:
        if option.mode == "remove":
            st.write(f"- **{option.mode}:** {option.reason}")
            label = newline_every_n_words(f"Mode: {option.mode} - {option.reason}", 12)

        if option.mode == "impute":
            st.write(f"- {option.mode}: {option.reason}")
            label = (
                f"**Mode:** {option.mode}, **Numerical strategy:** {option.numerical_strategy}, "
                f"**Categorical strategy:** {format_option(option.categorical_strategy)}\n"
                f"- {newline_every_n_words(option.reason, 12)}"
            )

        menu_options[label] = option

    choice = st.selectbox(
        "Select a missing value revision option:",
        list(menu_options.keys()) + ["Cancel"],
        index=None,
    )

    if choice is None:
        return PENDING

    if choice == "Cancel":
        st.info(
            "⇨ No revision has been selected for missing value handling. "
            "Previously selected settings will be used."
        )
        return None

    return menu_options[choice]


def version_tabs_ui(
    label: str = "Revision",
    key: str = "active_revision_tab",
    include_initial: bool = False,
    compare: bool = False,
) -> tuple[dict, Any | None]:
    """Create tabs for workflow versions (UI).

    Args:
        label (str, optional): Label used for each version tab. Defaults to "Revision".
        key (str, optional): Unique Streamlit key for the tab groups. Defaults to
            "active_revision_tab".
        include_initial (bool, optional): Whether to include version 1.
        compare (bool, optional): Whether to include a comparison tab.

    Returns:
        dict: Mapping of version IDs to their corresponding tabs.
    """
    versions = st.session_state.get("revision_versions", []).copy()

    if include_initial:
        versions.insert(0, 1)

    if not versions:
        return {}, None

    labels = [
        f"{label} {version - 1 if label == 'Revision' else version}"
        for version in versions
    ]

    if compare and len(versions) > 1:
        labels.append("Compare versions")

    tabs = st.tabs(labels, key=key, on_change="rerun",)

    version_tabs = dict(zip(versions, tabs[:len(versions)]))

    comparison_tab = tabs[-1] if compare and len(versions) > 1 else None

    return version_tabs, comparison_tab

from typing import Any, get_args

from langgraph.graph.state import CompiledStateGraph
import pandas as pd
import pyinputplus as pyip
import streamlit as st

from agentic_synth_data.prompts.generator_selection_prompt import generator_selection_message, \
    generator_review_message
from agentic_synth_data.schemas.generator_model import SelectGenerator
from agentic_synth_data.utils import GeneratorName, SemanticTypes, format_response, \
    info_helper_message, invoke_agent, output_default_parameters
from agentic_synth_data.preference_setup import SynthesisRequest


def generator_profile(dtype_df:pd.DataFrame) -> dict[str, Any]:
    """Prior to selecting the generator, view correlations and summary statistics 
    of numeric columns, which could help to inform data generator choice.

    Args:
        dtype_df (pd.DataFrame): The dataframe after dtype handling (dtype_handled_df).

    Returns:
        dict[str, Any]: A dictionary with the following information:
            - The shape of the dataframe (no row removal up to this point).
            - The number of unique values per column.
            - Numeric columns:
                - Summary statistics (including skew).
                - Correlation.
                - Correlation method.
    """
    # TODO: Add a warning about summary statistics for non local models.
    # TODO: Add more metrics for categorical columns.
    df = dtype_df.copy()
    col_profiles = {}
    shape = df.shape

    for col in df.columns:
        col_profiles[col] = {
            "n_unique": df[col].nunique(dropna=False),
        }

    num_df = df.select_dtypes(include="number")

    if num_df.empty:
        return {
            "shape": shape,
            "col_nunique": col_profiles,
            "num_summary": None,
            "num_corr": None,
            "corr_method": None,
        }

    # Select summary statistics (including skew)
    num_summary = num_df.describe()
    num_summary.loc["skew"] = num_df.skew().values

    print("Numeric value summary statistics:")
    print(num_summary)

    # Select correlations (Pearson or Spearman)
    # NOTE: could decide on the percentage of highly skewed columns
    if num_df.skew().abs().max() > 2:
        corr_method = "spearman"
    else:
        corr_method = "pearson"

    num_corr = num_df.corr(method=corr_method, numeric_only=True)

    print(f"\nCorrelation ({corr_method.capitalize()}):")
    print(num_corr)

    profile = {
        "shape": shape,
        "col_nunique": col_profiles,
        "num_summary": num_summary,
        "num_corr": num_corr,
        "corr_method": corr_method,
    }
    return profile


def generator_profile_ui(dtype_df:pd.DataFrame) -> dict[str, Any]:
    """Prior to selecting the generator, view correlations and summary statistics 
    of numeric columns, which could help to inform data generator choice (UI).

    Args:
        dtype_df (pd.DataFrame): The dataframe after dtype handling (dtype_handled_df).

    Returns:
        dict[str, Any]: A dictionary with the following information:
            - The shape of the dataframe (no row removal up to this point).
            - The number of unique values per column.
            - Numeric columns:
                - Summary statistics (including skew).
                - Correlation.
                - Correlation method.
    """
    # TODO: Add a warning about summary statistics for non local models.
    # TODO: Add more metrics for categorical columns.
    df = dtype_df.copy()
    col_profiles = {}
    shape = df.shape

    for col in df.columns:
        col_profiles[col] = {
            "n_unique": df[col].nunique(dropna=False),
        }

    num_df = df.select_dtypes(include="number")

    if num_df.empty:
        return {
            "shape": shape,
            "col_nunique": col_profiles,
            "num_summary": None,
            "num_corr": None,
            "corr_method": None,
        }

    # Select summary statistics (including skew)
    num_summary = num_df.describe()
    num_summary.loc["skew"] = num_df.skew().values

    st.subheader("Numeric value summary statistics:")
    st.write(num_summary)

    # Select correlations (Pearson or Spearman)
    # NOTE: could decide on the percentage of highly skewed columns
    if num_df.skew().abs().max() > 2:
        corr_method = "spearman"
    else:
        corr_method = "pearson"

    num_corr = num_df.corr(method=corr_method, numeric_only=True)

    st.subheader(f"\nCorrelation ({corr_method.capitalize()}):")
    st.write(num_corr)
    st.write("---")

    profile = {
        "shape": shape,
        "col_nunique": col_profiles,
        "num_summary": num_summary,
        "num_corr": num_corr,
        "corr_method": corr_method,
    }
    return profile


# TODO: Expand to parameter selection.
def select_generator(
    synthesis_request: SynthesisRequest,
    semantic_types: SemanticTypes,
    profile: dict[str, Any],
    agent: CompiledStateGraph,
    config: dict[str, dict[str, Any]],
) -> SynthesisRequest:
    """Based on the user's preferences and information about the dataset, the agent 
    selects a data generator mechanism.

    Args:
        synthesis_request (SynthesisRequest): Dictionary of the user's preferences.
        semantic_types (SemanticTypes): The semantic dtypes for each column in the dataset.
        agent (CompiledStateGraph): LangGraph agent used to recommend a mechanism/generator.
        config (dict[str, dict[str, Any]]: Configuration passed to the agent during invocation.

    Returns:
        SynthesisRequest: Synthesis request with updated generator information.                       
    """
    feedback_hist = []
    selected_mechanism = None
    choice_confirmed = False

    print()
    format_response(
        "Agent is coming up with a generator suggestion based on your preferences," \
        " the dataset's semantic types, the cardinality of the columns, and" \
        " potential summary statistics and correlations..."
        )

    while not choice_confirmed:
        message = generator_selection_message(
            synthesis_request=synthesis_request,
            semantic_types=semantic_types,
            profile=profile
        )

        if selected_mechanism is not None:
            message += generator_review_message(selected_mechanism)

        if feedback_hist:
            message += "\n\nUser feedback so far:\n" + "\n".join(feedback_hist)

        generation_plan = invoke_agent(
            prompt=message, model_cls=SelectGenerator, agent=agent, config=config,
        )

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
            synthesis_request.generator.name = generation_plan.generator
            synthesis_request.generator.parameters = output_default_parameters(
                synthesis_request.generator.name
            )
            choice_confirmed = True
            print(f"\n=> {generation_plan.generator} has been approved.")

        elif decision == "choose another":
            selected_mechanism = pyip.inputMenu(
                list(get_args(GeneratorName.__value__)),
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


def select_generator_ui(
    synthesis_request: SynthesisRequest,
    semantic_types: SemanticTypes,
    profile: dict[str, Any],
    agent: CompiledStateGraph,
    config: dict[str, dict[str, Any]],
) -> SynthesisRequest:
    """Based on the user's preferences and information about the dataset, the agent 
    selects a data generator mechanism (UI).

    Args:
        synthesis_request (SynthesisRequest): Dictionary of the user's preferences.
        semantic_types (SemanticTypes): The semantic dtypes for each column in the dataset.
        agent (CompiledStateGraph): LangGraph agent used to recommend a mechanism/generator.
        config (dict[str, dict[str, Any]]: Configuration passed to the agent during invocation.

    Returns:
        SynthesisRequest: Synthesis request with updated generator information.                       
    """
    if "generator_feedback_hist" not in st.session_state:
        st.session_state.generator_feedback_hist = []

    if "generator_plan" not in st.session_state:
        message = generator_selection_message(
            synthesis_request=synthesis_request,
            semantic_types=semantic_types,
            profile=profile,
        )

        selected_mechanism = st.session_state.get("pending_generator_review")

        if selected_mechanism is not None:
            message += generator_review_message(selected_mechanism)

        feedback_hist = st.session_state.generator_feedback_hist

        if feedback_hist:
            message += "\n\nUser feedback so far:\n" + "\n".join(
                f"{feedback}" for feedback in feedback_hist
            )

        with st.spinner("Agent is coming up with a generator suggestion..."):
            generation_plan = invoke_agent(
                prompt=message, model_cls=SelectGenerator, agent=agent, config=config, ui=True,
            )

        st.session_state.generator_plan = generation_plan

        st.session_state.pending_generator_review = None

    else:
        generation_plan = st.session_state.generator_plan

    info_helper_message(
        message="Agent generator evaluation completed.",
        context=(
            "The agent created a generator suggestion based on:\n"
            "- Your preferences.\n"
            "- The dataset's semantic types.\n"
            "- The cardinality of the columns.\n"
            "Potential summary statistics and correlations."
        ),
        key="generator_selection_info",
    )

    st.subheader("Suggestion:")
    st.write(f"Data generator: {generation_plan.generator}")

    st.subheader("Pros:")
    st.write(generation_plan.pros)

    st.subheader("Cons:")
    st.write(generation_plan.cons)

    st.subheader("Rationale:")
    st.write(generation_plan.rationale)

    if len(generation_plan.warning) > 0:
        st.warning(f"Warning: {generation_plan.warning[0]}")

    if st.session_state.get("generator_selection_complete", False,):
        st.info(f"=> {generation_plan.generator} has been approved.")
        return st.session_state.approved_synthesis_request

    st.header("Do you approve using the suggested generator?")
    decision = st.selectbox(
        "Do you approve using the suggested generator?",
        ["approve", "choose another", "discuss",],
        index=None,
        label_visibility="hidden",
    )

    if decision is None:
        st.stop()

    if decision == "approve":
        synthesis_request.generator.name = generation_plan.generator
        synthesis_request.generator.parameters = output_default_parameters(
            synthesis_request.generator.name
        )

        st.session_state.approved_synthesis_request = synthesis_request
        st.session_state.generator_selection_complete = True

        st.info(f"=> {generation_plan.generator} has been approved.")
        return synthesis_request

    if decision == "choose another":
        st.write("Enter generator to evaluate:")
        selected_mechanism = st.selectbox(
            "select another mechanism",
            list(get_args(GeneratorName.__value__)),
            index=None,
            label_visibility="hidden",
        )

        if selected_mechanism is None:
            st.stop()

        if st.button( # Not sure whether to keep this button (with or without button)
            "Evaluate generator",
            key="evaluate_generator_button",
        ):
            st.session_state.pending_generator_review = selected_mechanism

            st.session_state.pop("generator_plan", None)

            st.rerun()

        st.stop()

    else: # discuss
        st.write("Write feedback or questions about the generator choice")

        feedback = st.chat_input(
            "Ask about the generator recommendation.",
            key="generator_feedback_input",
        )

        if feedback:
            st.session_state.generator_feedback_hist.append(feedback.strip())
            st.session_state.pop("generator_plan", None)
            st.rerun()

        st.stop()

    return None

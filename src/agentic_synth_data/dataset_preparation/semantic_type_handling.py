from typing import Any, Literal

import pandas as pd
import numpy as np
from pydantic import RootModel
import pyinputplus as pyip
from langgraph.graph.state import CompiledStateGraph
import streamlit as st

from agentic_synth_data.prompts.stype_prompt import semantic_type_message
from agentic_synth_data.utils import df_formatter_st, extract_output, info_helper_message, \
    parse_json, parse_json_ui


class STypeInference(RootModel[dict[str, Literal["Float", "Integer", "Boolean", "Categorical", "Ordinal",]]]):
    pass


def semantic_type_profile(df:pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Create a profile for each column in the dataframe for the agent to use in 
    inferring the semantic types of each column.

    Args:
        df (pd.DataFrame): The input dataframe with corrected dtypes.

    Returns:
        dict[str, dict[str, Any]: Semantic type profile where keys are column names
        and values are dictionaries which hold information specific to the column:
        - dtype: Python data type (str).
        - unique_values: Contains unique values and their counts (dict[str, int]).
        - n_unique: The number of unique values (int).
    """
    profiles = {}

    for col in df.columns:
        profiles[col] = {
            "dtype": str(df[col].dtype),
            "unique_values": df[col].value_counts(dropna=False).head(15).to_dict(),
            "n_unique": df[col].nunique(dropna=False),
        }
    return profiles

# Semantic type dict type channge to literal
def handle_semantic_types(df:pd.DataFrame, agent:CompiledStateGraph, config:dict) -> dict[str, Literal["Float", "Integer", "Boolean", "Categorical", "Ordinal",]]:
    """Determine the semantic type of each column in a dataframe by:
    1. Making a profile for each column (dtype, unique non null values and 
       their counts, and total number of non null unique values).
    2. Agent infers the semantic type of columns based on their profile.
    3. User confirms or denies correctness of the agent's response.
    4. If corrections are needed, the user can manually update column types.

    Args:
        df (pd.DataFrame): The input dataframe.
        agent (CompiledStateGraph): LangGraph agent used to infer column semantic types.
        config (dict): Configuration passed to the agent during invocation.

    Returns:
        dict[str, str]: Semantic types (value) assigned to each column (key).
    """
    # 1. Make a profile of the semantic types of each column.
    profiles = semantic_type_profile(df)
    prompt = semantic_type_message(profiles)

    # 2. Agent infers the semantic type of columns based on their profile.
    print(
        "\nAgent is analyzing column data types, unique values, and value distributions to " \
        "infer semantic types..."
    )
    raw_inference = agent.invoke(
        {"messages": [{"role": "user","content": prompt,}]},
        config=config,
    )

    plan = extract_output(raw_inference)

    semantic_types = parse_json(
        plan=plan, 
        model_cls=STypeInference,
        agent=agent,
        config=config,
        root=True
    ) # TODO: check more extensively

    # TODO:Include more formal checks that each column is accounted for:
    # - Output must match existing column names
    # - Existing column count must match
    # NOTE: No issues have been observed with this, but good to keep an eye out for.
    print("\nDetected semantic types for each column:")
    print("Note: Supported types are Integer, Float, Boolean, Categorical, and Ordinal.")
    print()
    for col, stype in semantic_types.items():
        print(f"{col}: {stype}")

    # 3. User confirmation.
    confirm = pyip.inputYesNo("\nAre all column data types correct?")
    # possible TODO: Add a way to select multiple columns as a group and to change their stype (semantic type).

    # 4. If corrections are needed, the user can manually update column types.
    if confirm == "no":
        while True:
            print(
                "\nInput column name/number to modify (enter 'exit' (not case-sensitive) or 1 to" \
                " finish)"
            )
            col = pyip.inputMenu(["Exit"] + df.columns.tolist(), numbered=True)

            if col == "Exit":
                break

            print(f"\nSelected the following column: {col}")
            print(
                f"Current semantic type is '{semantic_types[col]}'. What should this be modified" \
                " to?"
            )

            choice = pyip.inputMenu(
                ["Integer", "Float", "Boolean", "Categorical", "Ordinal"],
                numbered=True
            )

            semantic_types[col] = choice

            print(f"\n=> Updated semantic type for {col}: {semantic_types[col]}")
            print("For reference: Semantic types present in each column:")
            for col, stype in semantic_types.items():
                print(f"{col}: {stype}")
            print("-"*40)
    print()
    print("-"*25+"Final data types present in each column"+"-"*25)
    for col, stype in semantic_types.items():
        print(f"{col}: {stype}")
    return semantic_types


def handle_semantic_types_ui(df:pd.DataFrame, agent:CompiledStateGraph, config:dict) -> dict[str, Literal["Float", "Integer", "Boolean", "Categorical", "Ordinal",]]:
    """Determine the semantic type of each column in a dataframe by (UI):
    1. Making a profile for each column (dtype, unique non null values and 
       their counts, and total number of non null unique values).
    2. Agent infers the semantic type of columns based on their profile.
    3. User confirms or denies correctness of the agent's response.
    4. If corrections are needed, the user can manually update column types.

    Args:
        df (pd.DataFrame): The input dataframe.
        agent (CompiledStateGraph): LangGraph agent used to infer column semantic types.
        config (dict): Configuration passed to the agent during invocation.

    Returns:
        dict[str, str]: Semantic types (value) assigned to each column (key).
    """
    # 1. Generate agent inference once, then save it.
    if "semantic_types_inference" not in st.session_state:
        # Make a profile of the semantic types of each column.
        profiles = semantic_type_profile(df)
        prompt = semantic_type_message(profiles)

        # Agent infers the semantic type of columns based on their profile.
        with st.spinner("Agent is analyzing dataset characteristics to infer semantic types..."):
            raw_inference = agent.invoke(
                {"messages": [{"role": "user","content": prompt,}]},
                config=config,
            )

            plan = extract_output(raw_inference)

            inferred_semantic_types = parse_json_ui(
                plan=plan,
                model_cls=STypeInference,
                agent=agent,
                config=config,
                root=True,
            )

        st.session_state.semantic_types_inference = inferred_semantic_types.copy()

    # 2. Reuse and display saved agent inference.
    else:
        inferred_semantic_types = st.session_state.semantic_types_inference.copy()

    info_helper_message(
        message="Agent semantic type inference completed.",
        context=(
            "The agent inferred semantic types based on each column's:\n- Assigned data type.\n- "
            "Present unique values.\n- Distribution of unique values."
        ),
        key="semantic_type_info",
    )

    st.write("Detected semantic types for each column:")

    stype_table = pd.DataFrame({
        "Column": inferred_semantic_types.keys(),
        "Semantic type": inferred_semantic_types.values(),
    })

    df_formatter_st(stype_table)

    # 3. If already completed, return the final saved result.
    if st.session_state.get("semantic_types_complete", False):
        semantic_types = st.session_state.semantic_types.copy()

        if semantic_types != inferred_semantic_types:
            st.subheader("Final semantic types for each column:")

            final_stype_table = pd.DataFrame({
                "Column": semantic_types.keys(),
                "Semantic type": semantic_types.values(),
            })

            df_formatter_st(final_stype_table)

        return semantic_types

    # 4. User confirmation.
    confirm = st.selectbox(
        "Are all semantic types correct?",
        ["yes", "no"],
        index=None,
        help=(
            "Supported types are Integer, Float, Boolean, Categorical, and Ordinal.\n\n"
            "Examples:\n"
            "- Integer: whole number values, e.g. 0, 3, 12, -5\n"
            "- Float: continuous numerical values, e.g. 2.5, -0.7, 14.32\n"
            "- Boolean: binary values, e.g. True/False, Yes/No\n"
            "- Categorical: unordered labels or groups, e.g. 'red', 'blue', 'green'\n"
            "- Ordinal: categories with a meaningful order, e.g. 'low', 'medium', 'high'"
        ),
    )

    if confirm is None:
        st.stop()

    semantic_types = inferred_semantic_types.copy()

    # 5. If corrections are needed, the user can manually update column types.
    if confirm == "no":
        st.write("Adjust any incorrect semantic types below.")

        with st.form("semantic_type_correction_form"):
            semantic_type_choices = {}
            options = ["Integer", "Float", "Boolean", "Categorical", "Ordinal",]

            for col in df.columns:
                current_type = semantic_types[col]
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(col)

                with col2:
                    st.write(f"Current: **{current_type}**")

                with col3:
                    semantic_type_choices[col] = st.selectbox(
                        "New semantic type",
                        options,
                        index=options.index(current_type),
                        key=f"semantic_type_{col}",
                        label_visibility="collapsed",
                    )

            submitted = st.form_submit_button("Apply changes")

        if not submitted:
            st.stop()

        for col, choice in semantic_type_choices.items():
            semantic_types[col] = choice

    st.session_state.semantic_types = semantic_types
    st.session_state.semantic_types_complete = True
    return semantic_types


def prepare_for_syndat(
        real_df: pd.DataFrame, synthetic_df: pd.DataFrame, semantic_types: dict[str, str]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match real and synthetic dataframe columns to previously defined dtypes for compatibility 
    with Syndat.
    Args:
        real_df (pd.DataFrame): The real dataframe (either with missing values replaced or not).
        synthetic_df (pd.DataFrame): The synthetic dataframe.
        semantic_types (dict[str, str]): Dictionary with the semantic types of each column.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
        - pd.DataFrame: The real dataframe with prepared dtypes.
        - pd.DataFrame: The synthetic dataframe with prepared dtypes.
    """
    real = real_df.copy()
    synth = synthetic_df.copy()

    # Same columns/order
    synth = synth[real.columns]

    for col in real.columns:
        stype = semantic_types[col]

        if stype in ["Categorical", "Ordinal", "Boolean"]:
            real[col] = real[col].astype("object")
            synth[col] = synth[col].astype("object")

            real[col] = real[col].where(~real[col].isna(), np.nan)
            synth[col] = synth[col].where(~synth[col].isna(), np.nan)

        elif stype == "Integer":
            real[col] = pd.to_numeric(real[col], errors="coerce")
            synth[col] = pd.to_numeric(synth[col], errors="coerce")

        elif stype == "Float":
            real[col] = pd.to_numeric(real[col], errors="coerce")
            synth[col] = pd.to_numeric(synth[col], errors="coerce")

    return real, synth

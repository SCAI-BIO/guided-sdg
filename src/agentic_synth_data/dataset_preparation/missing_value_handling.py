from typing import Optional, Any

import pandas as pd
import numpy as np
import pyinputplus as pyip
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer
import streamlit as st

from agentic_synth_data.utils import display_synthesis_request_st, format_response, \
    extract_output, info_helper_message, parse_json, display_synthesis_request, parse_json_ui
from agentic_synth_data.prompts.missing_value_prompt import missing_value_prompt, missing_feedback
from agentic_synth_data.preference_setup import SynthesisRequest, ImputationConfig


class MissingHandling(BaseModel):
    mode: str
    set_threshold: int | None = None
    imputation_method: str | None = None
    categorical_strategy: str | None = None
    numerical_strategy: str | None = None
    warning: list[str] = Field(default_factory=list)
    rationale: str


def update_missing_value_choice(synthesis_request:SynthesisRequest) -> SynthesisRequest:
    """Update missing value handling choice in the case that either the remove all or
     threshold setting for null values is too harsh.

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.

    Returns:
        SynthesisRequest: Updated synthesis request.
    """
    choice = pyip.inputMenu(
        ["auto", "impute", "threshold"],
        "\nPlease select another method to handle missing values:\n",
        numbered=True
    )
    synthesis_request.missing_values.mode = choice

    if choice == "threshold":
        value = pyip.inputInt(
            "\nSet the maximum number of missing values allowed per row:\n",
            min=0,
        )
        synthesis_request.missing_values.threshold = value
        synthesis_request.missing_values.impute = ImputationConfig()

    elif choice == "impute":
        synthesis_request.missing_values.impute.method = "SimpleImputer"
        synthesis_request.missing_values.threshold = None

        cat_choice = pyip.inputMenu(
            ["auto", "most frequent", "constant"],
            "\nHow would you like to handle missing categorical values? Select 'auto' if you" \
            " would like a suggestion.\n", 
            numbered=True,
        )

        if cat_choice == "most frequent":
            cat_choice = "most_frequent"

        synthesis_request.missing_values.impute.categorical_strategy = cat_choice

        num_choice = pyip.inputMenu(
            ["auto", "mean", "median", "most frequent"],
            "\nHow would you like to handle missing numerical values? Select 'auto' if you would" \
            " like a suggestion.\n", 
            numbered=True,
        )

        if num_choice == "most frequent":
            num_choice = "most_frequent"

        synthesis_request.missing_values.impute.numerical_strategy = num_choice

    elif choice == "auto":
        synthesis_request.missing_values.threshold = None
        synthesis_request.missing_values.impute = ImputationConfig()

    return synthesis_request


def update_missing_value_choice_ui(synthesis_request:SynthesisRequest) -> SynthesisRequest:
    """Update missing value handling choice in the case that either the remove all or
    threshold setting for null values is too harsh (UI).

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.

    Returns:
        SynthesisRequest: Updated synthesis request.
    """
    st.write("Please select another method to handle missing values:")
    choice = st.selectbox(
        label="Update missing value handling choice",
        options=["auto", "impute",],
        index=None,
        key="update_missing_value_handling_choice",
    )

    if choice is None:
        st.stop()

    synthesis_request.missing_values.mode = choice

    if choice == "impute":
        synthesis_request.missing_values.impute.method = "SimpleImputer"
        synthesis_request.missing_values.threshold = None

        st.write(
            "How would you like to handle missing categorical values? Select 'auto' if you would "
            "like a suggestion."
        )

        cat_choice = st.selectbox(
            label="Update missing value handling choice",
            options=["auto", "impute",],
            index=None,
            key="update_missing_value_handling_choice",
        )

        if cat_choice == "most frequent":
            cat_choice = "most_frequent"

        synthesis_request.missing_values.impute.categorical_strategy = cat_choice

        num_choice = st.selectbox(
            label="Update missing value handling choice",
            options=["auto", "impute",],
            index=None,
            key="update_missing_value_handling_choice",
        )

        if num_choice == "most frequent":
            num_choice = "most_frequent"

        synthesis_request.missing_values.impute.numerical_strategy = num_choice

    elif choice == "auto":
        synthesis_request.missing_values.threshold = None
        synthesis_request.missing_values.impute = ImputationConfig()

    return synthesis_request


def auto_update_missing_plan(synthesis_request:SynthesisRequest, missing_plan:dict[str, Any]) -> SynthesisRequest:
    """When "auto" is selected in the missing value handling plan, the agent is called and 
    returns a structured dictionary. This structured dictionary is changed to the format of
    the synthesis request.

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        missing_plan (dict[str, Any]): Agent recommendation for handling missing values.

    Returns:
        SynthesisRequest: Updated synthesis request.
    """
    miss = synthesis_request.missing_values

    miss.mode = missing_plan["mode"]
    miss.impute.method = missing_plan["imputation_method"]
    miss.impute.categorical_strategy = missing_plan["categorical_strategy"]
    miss.impute.numerical_strategy = missing_plan["numerical_strategy"]
    miss.threshold = missing_plan["set_threshold"]
    return synthesis_request


def handle_missing_values(
        df:pd.DataFrame,
        synthesis_request:SynthesisRequest,
        semantic_types:dict[str, str],
        agent:Optional[CompiledStateGraph],# | None = None,
        config:dict,# | None = None,
        ) -> tuple[pd.DataFrame, SynthesisRequest] | None:
    """Handle missing values in a dataframe based on user defined preferences.

    Missing value strategies include:
    1. Remove all rows containing null values.
    2. Remove rows exceeding a specified threshold for null values.
    3. Imputation.
    4. Receive a suggestion from an agent on how to handle null values.

    Args:
        df (pd.DataFrame): The input dataframe.
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        semantic_types (dict[str, str]): Map of column names to semantic types (e.g., categorical, 
                               ordinal, integer, float).
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to 
                               recommend a strategy for missing values. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation.
                               Defaults to None.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]] | None: 
        - (pd.DataFrame) The processed dataframe after missing value handling. 
        - (SynthesisRequest) The updated synthesis_request.
        None: Agent planned missing value handling, approved by user.
    """
    if df.isna().sum().sum() == 0:
        print(
            "\nThere are no missing values in the dataset. Continuing to synthetic data generation."
        )
        return df, synthesis_request

    print(f"\n{df.isna().sum().sum()} samples with missing values detected in the dataset.")
    mode = synthesis_request.missing_values.mode
    impute = synthesis_request.missing_values.impute

    # 1. Remove all samples with NA columns.
    if mode == "remove":
        null_row_count = df.isna().any(axis=1).sum()
        print(
            f"\nWarning: {null_row_count} out of {df.shape[0]} rows (" \
            f"{null_row_count/df.shape[0]*100}%) will be removed from the dataset."
        )
        confirm = pyip.inputYesNo("\nConfirm whether this should be done (yes/no/y/n):\n")
        if confirm == "yes":
            df = df.dropna()
            print(
                "\n=> All samples with missing values have been removed. There are " \
                f"{df.isna().sum().sum()} remaining missing values."
            )
            return df, synthesis_request

        # If this option is too harsh and the user wants to try another method.
        synthesis_request = update_missing_value_choice(synthesis_request)
        return handle_missing_values(df, synthesis_request, semantic_types, agent, config)

    # 2. Remove all samples above a specified threshold of NA columns. 
    if mode == "threshold":
        threshold_samples = df.isna().sum(axis=1) > synthesis_request.missing_values.threshold
        count = threshold_samples.sum()

        print(
            f"Warning: {count} out of {df.shape[0]} ({count/df.shape[0]*100}%) will be removed " \
            "from the dataset."
        )

        confirm = pyip.inputYesNo("\nConfirm whether this should be done.\n")
        if confirm == "yes":
            df = df[~threshold_samples]
            return df, synthesis_request

        confirm = pyip.inputYesNo("\nWould you like to select a new threshold value?\n")
        if confirm == "yes":
            value = pyip.inputInt("\nEnter a new threshold value.\n", min=1) # TODO: Double check how I have it written
            synthesis_request.missing_values.threshold = value

            return handle_missing_values(df, synthesis_request, semantic_types, agent, config)

        synthesis_request = update_missing_value_choice(synthesis_request)
        # TODO add a limit to the threshold such that it doesn't add more
        #  possible TODO allow for threshold changes (if found to be too harsh but want to stay with a threshold)

        return handle_missing_values(df, synthesis_request, semantic_types, agent, config)

    # 3. Perform imputation given there are decided categorical and numerical strategies.
    if (
        mode == "impute" and
        impute.categorical_strategy != "auto" and
        impute.numerical_strategy != "auto"
    ):
        # Separate user defined categorical and numerical values for imputation
        grouped = {}
        for col, stype in semantic_types.items():
            grouped.setdefault(stype, []).append(col)

        cat_cols = grouped.get("Categorical", []) + grouped.get("Ordinal", [])
        num_cols = grouped.get("Integer", []) + grouped.get("Float", [])

        df = df.copy()
        print("\nReview of missing value handling results:")
        # Categorical value imputation
        cat_strategy = synthesis_request.missing_values.impute.categorical_strategy
        if cat_cols:
            cat_data = df[cat_cols].astype("object").where(df[cat_cols].notna(), np.nan)

            if cat_strategy == "constant":
                print(
                    "\nEnter the fill value that missing values should have. Fill values " \
                    "must be strings.\nPress enter if you would like to stay with the " \
                    "default ('Unknown'):"
                )
                fill_value = pyip.inputStr()

                if fill_value == "":
                    fill_value = "Unknown"

                imputer = SimpleImputer(strategy=cat_strategy, fill_value=fill_value)
                print(f"Missing values filled with '{fill_value}'")

            else:
                imputer = SimpleImputer(strategy=cat_strategy)

            df[cat_cols] = pd.DataFrame(
                imputer.fit_transform(cat_data),
                columns=cat_cols,
                index=df.index
            )

            print(
                f"- There are {df[cat_cols].isna().sum().sum()} remaining null values" \
                " in the categorical columns."
            )

        # Numerical value imputation
        if num_cols:
            num_strategy = synthesis_request.missing_values.impute.numerical_strategy
            imputer = SimpleImputer(strategy=num_strategy)

            df[num_cols] = pd.DataFrame(
                imputer.fit_transform(df[num_cols]), 
                columns=df[num_cols].columns, 
                index=df.index
            )

            print(
                f"- There are {df[num_cols].isna().sum().sum()} remaining null values" \
                " in the numerical columns."
            )

        print(f"\n=> There are {df.isna().sum().sum()} remaining null values in the dataset.")
        return df, synthesis_request

    # 4. Use the agent to suggest the missing value handling strategy.
    if (
        mode == "auto" or 
        impute.categorical_strategy == "auto" or 
        impute.numerical_strategy == "auto"
    ):
        proportion_missing = (df.isna().sum() / df.shape[0]).round(4).to_dict()
        rows_with_missing = int(df.isna().any(axis=1).sum())
        total_rows = df.shape[0]

        request_dict = synthesis_request.model_dump()

        prompt = missing_value_prompt(
            proportion_missing=proportion_missing, 
            rows_with_missing=rows_with_missing, 
            total_rows=total_rows, 
            synthesis_request=request_dict, 
            semantic_types=semantic_types, 
            )

        print("\nAgent is coming up with a suggestion for missing value handling...")
        feedback = None
        while True:
            if feedback is None:
                current_prompt = prompt
            else:
                current_prompt = missing_feedback(prompt, feedback)
                print("Waiting for agent response...")

            raw_plan = agent.invoke(
                {"messages": [{"role": "user", "content": current_prompt,}]},
                config=config,
            )

            plan = extract_output(raw_plan)

            missing_plan = parse_json(
                plan=plan, model_cls=MissingHandling, agent=agent, config=config,
            )

            updated_plan = {
                "mode": missing_plan.mode,
                "imputation_method": missing_plan.imputation_method,
                "categorical_strategy": missing_plan.categorical_strategy, 
                "numerical_strategy": missing_plan.numerical_strategy,
                "set_threshold": missing_plan.set_threshold,
                "warning": missing_plan.warning,
                "rationale": missing_plan.rationale,
            }

            print("\nSuggested plan:")
            print(f"- Mode: {updated_plan['mode']}")
            if updated_plan["set_threshold"] is not None:
                print(f"- Threshold: {updated_plan['set_threshold']}")
            if updated_plan["imputation_method"] is not None:
                print(f"- Imputation method: {updated_plan['imputation_method']}")
            if updated_plan["categorical_strategy"] is not None:
                print(f"- Categorical strategy: {updated_plan['categorical_strategy']}")
            if updated_plan["numerical_strategy"]:
                print(f"- Numerical strategy: {updated_plan['numerical_strategy']}")

            if len(updated_plan["warning"]) > 0:
                print("\nWarning:")
                format_response(f"- Warning: {updated_plan['warning'][0]}")

            print(f"\nReasoning behind the suggestion:")
            format_response(updated_plan["rationale"])

            confirm = pyip.inputYesNo("\nWould you like to proceed with the suggestion?")
            if confirm == "yes":
                # Update synthesis request with new specifications
                synthesis_request = auto_update_missing_plan(synthesis_request, updated_plan)
                print(f"\n\nThe missing value handling plan has been updated to the following:")
                display_synthesis_request(synthesis_request)
                break

            print(
                "\n\nPlease describe how you would want missing values to be" \
                " handled and what changes you would make to the suggested plan."
            )

            feedback = input()

        # Update synthesis request with new specifications
        synthesis_request = auto_update_missing_plan(synthesis_request, updated_plan)
        return handle_missing_values(df, synthesis_request, semantic_types, agent, config)


def handle_missing_values_ui(
        df:pd.DataFrame,
        synthesis_request:SynthesisRequest,
        semantic_types:dict[str, str],
        agent:Optional[CompiledStateGraph],
        config:dict,
        ) -> tuple[pd.DataFrame, SynthesisRequest] | None:
    """Handle missing values in a dataframe based on user defined preferences (UI).

    Missing value strategies include:
    1. Remove all rows containing null values.
    2. Remove rows exceeding a specified threshold for null values.
    3. Imputation.
    4. Receive a suggestion from an agent on how to handle null values.

    Args:
        df (pd.DataFrame): The input dataframe.
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        semantic_types (dict[str, str]): Map of column names to semantic types (e.g., categorical, 
                               ordinal, integer, float).
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to 
                               recommend a strategy for missing values. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation.
                               Defaults to None.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]] | None: 
        - (pd.DataFrame) The processed dataframe after missing value handling. 
        - (SynthesisRequest) The updated synthesis_request.
        None: Agent planned missing value handling, approved by user.
    """
    if df.isna().sum().sum() == 0:
        st.write(
            "\nThere are no missing values in the dataset. Continuing to synthetic data generation."
        )
        return df, synthesis_request

    st.write(f"\n{df.isna().sum().sum()} samples with missing values detected in the dataset.")
    mode = synthesis_request.missing_values.mode
    impute = synthesis_request.missing_values.impute

    # 1. Remove all samples with NA columns.
    if mode == "remove":
        null_row_count = df.isna().any(axis=1).sum()
        st.warning(
            f"\nWarning: {null_row_count} out of {df.shape[0]} rows ("
            f"{null_row_count/df.shape[0]*100}%) will be removed from the dataset."
        )
        st.write("\nConfirm whether this should be done:\n")
        confirm = st.selectbox(
            "removal of missing values yes/no",
            ["yes", "no"],
            index=None,
            label_visibility="hidden",
        )

        if confirm is None:
            st.stop()

        if confirm == "yes":
            df = df.dropna()
            st.info(
                "\n=> All samples with missing values have been removed. There are "
                f"{df.isna().sum().sum()} remaining missing values."
            )
            return df, synthesis_request

        # If this option is too harsh and the user wants to try another method.
        synthesis_request = update_missing_value_choice_ui(synthesis_request)
        return handle_missing_values(df, synthesis_request, semantic_types, agent, config)

    # 3. Perform imputation given there are decided categorical and numerical strategies.
    if (
        mode == "impute" and
        impute.categorical_strategy != "auto" and
        impute.numerical_strategy != "auto"
    ):
        # Separate user defined categorical and numerical values for imputation
        grouped = {}
        for col, stype in semantic_types.items():
            grouped.setdefault(stype, []).append(col)

        cat_cols = grouped.get("Categorical", []) + grouped.get("Ordinal", [])
        num_cols = grouped.get("Integer", []) + grouped.get("Float", [])

        df = df.copy()
        st.write("\nReview of missing value handling results:")
        # Categorical value imputation
        cat_strategy = synthesis_request.missing_values.impute.categorical_strategy
        if cat_cols:
            cat_data = df[cat_cols].astype("object").where(df[cat_cols].notna(), np.nan)

            if cat_strategy == "constant":
                st.write(
                    "\nEnter the fill value that missing values should have. Fill values "
                    "must be strings.\nPress enter if you would like to stay with the "
                    "default ('Unknown'):"
                )
                fill_value = st.text_input("fill_value")

                if fill_value == "":
                    fill_value = "Unknown"

                imputer = SimpleImputer(strategy=cat_strategy, fill_value=fill_value)
                st.write(f"Missing values filled with '{fill_value}'")

            else:
                imputer = SimpleImputer(strategy=cat_strategy)

            df[cat_cols] = pd.DataFrame(
                imputer.fit_transform(cat_data),
                columns=cat_cols,
                index=df.index
            )

            st.write(
                f"- There are {df[cat_cols].isna().sum().sum()} remaining null values"
                " in the categorical columns."
            )

        # Numerical value imputation
        if num_cols:
            num_strategy = synthesis_request.missing_values.impute.numerical_strategy
            imputer = SimpleImputer(strategy=num_strategy)

            df[num_cols] = pd.DataFrame(
                imputer.fit_transform(df[num_cols]),
                columns=df[num_cols].columns,
                index=df.index
            )

            st.write(
                f"- There are {df[num_cols].isna().sum().sum()} remaining null values"
                " in the numerical columns."
            )

        st.write(f"\n=> There are {df.isna().sum().sum()} remaining null values in the dataset.")
        return df, synthesis_request

    # 4. Use the agent to suggest the missing value handling strategy.
    if (
        mode == "auto" or
        impute.categorical_strategy == "auto" or
        impute.numerical_strategy == "auto"
    ):
        proportion_missing = (df.isna().sum() / df.shape[0]).round(4).to_dict()
        rows_with_missing = int(df.isna().any(axis=1).sum())
        total_rows = df.shape[0]

        request_dict = synthesis_request.model_dump()

        prompt = missing_value_prompt(
            proportion_missing=proportion_missing,
            rows_with_missing=rows_with_missing,
            total_rows=total_rows,
            synthesis_request=request_dict,
            semantic_types=semantic_types,
        )

        if "missing_plan" not in st.session_state:
            with st.spinner("Agent is coming up with a suggestion for missing value handling..."):
                raw_plan = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt,}]},
                    config=config,
                )

                plan = extract_output(raw_plan)

                missing_plan = parse_json_ui(
                    plan=plan, model_cls=MissingHandling, agent=agent, config=config,
                )


                st.session_state.missing_plan = {
                    "mode": missing_plan.mode,
                    "imputation_method": missing_plan.imputation_method,
                    "categorical_strategy": missing_plan.categorical_strategy, 
                    "numerical_strategy": missing_plan.numerical_strategy,
                    "set_threshold": missing_plan.set_threshold,
                    "warning": missing_plan.warning,
                    "rationale": missing_plan.rationale,
                }


            updated_plan = st.session_state.missing_plan

            info_helper_message(
                message="Agent has finished coming up with suggestion.",
                context=(
                    "The agent suggested a missing value handling strategy based on:\n- The "
                    "proportion of missing values for each column.\n- The number of rows with "
                    "missing values.\n- Total rows in the dataset.\n- Your synthesis request.\n- "
                    "The semantic data types of each column present in the dataset."
                ),
                key="missing_handling_info",
            )

            st.write("Suggested plan:")
            st.write(f"- Mode: {updated_plan['mode']}")

            if updated_plan["set_threshold"] is not None:
                st.write(f"- Threshold: {updated_plan['set_threshold']}")

            if updated_plan["imputation_method"] is not None:
                st.write(f"- Imputation method: {updated_plan['imputation_method']}")

            if updated_plan["categorical_strategy"] is not None:
                st.write(f"- Categorical strategy: {updated_plan['categorical_strategy']}")

            if updated_plan["numerical_strategy"]:
                st.write(f"- Numerical strategy: {updated_plan['numerical_strategy']}")

            if len(updated_plan["warning"]) > 0:
                st.write("Warning:")
                st.warning(f"- Warning: {updated_plan['warning'][0]}")

            st.write("Rationale behind the suggestion:")
            st.write(updated_plan["rationale"])

            confirm = st.selectbox(
                "Would you like to proceed with the suggestion?",
                ["yes", "no"],
                index=None,
                key="missing_handling_check_suggestion",
            )

            if confirm == "yes":
                # Update synthesis request with new specifications
                synthesis_request = auto_update_missing_plan(synthesis_request, updated_plan)

                st.subheader(
                    "The missing value handling plan has been updated to the following:"
                )
                display_synthesis_request_st(synthesis_request)

            elif confirm == "no":
                st.write(
                    "Please describe how you would want missing values to be "
                    "handled and what changes you would make to the suggested plan."
                )

                feedback = st.text_area(
                    "Enter your feedback",
                    key="missing_handling_feedback",
                )

                if st.button("Submit feedback", key="missing_handling_feedback_submit",):
                    current_prompt = missing_feedback(prompt, feedback)

                    with st.spinner("Agent is updating suggestion..."):
                        raw_plan = agent.invoke(
                            {"messages": [{"role": "user", "content": current_prompt,}]},
                            config=config,
                        )

                        plan = extract_output(raw_plan)

                        missing_plan = parse_json_ui(
                            plan=plan, model_cls=MissingHandling, agent=agent, config=config,
                        )

                    st.session_state.missing_plan = {
                        "mode": missing_plan.mode,
                        "imputation_method": missing_plan.imputation_method,
                        "categorical_strategy": missing_plan.categorical_strategy, 
                        "numerical_strategy": missing_plan.numerical_strategy,
                        "set_threshold": missing_plan.set_threshold,
                        "warning": missing_plan.warning,
                        "rationale": missing_plan.rationale,
                    }

                st.session_state.missing_handling_check_suggestion = None
                st.rerun()

        # Update synthesis request with new specifications
        synthesis_request = auto_update_missing_plan(synthesis_request, updated_plan)
        return handle_missing_values(df, synthesis_request, semantic_types, agent, config)

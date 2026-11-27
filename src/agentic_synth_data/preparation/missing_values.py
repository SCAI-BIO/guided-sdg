from typing import Optional, Any, get_args

from langchain_core.runnables import RunnableConfig
import pandas as pd
import numpy as np
import pyinputplus as pyip
from langgraph.graph.state import CompiledStateGraph
from sklearn.impute import SimpleImputer
import streamlit as st

from agentic_synth_data.utils import CategoricalChoiceAuto, NumericalChoiceAuto, format_option, \
    format_response, info_helper_message, invoke_agent, display_synthesis_request, SemanticTypes
from agentic_synth_data.prompts.missing_value import missing_value_prompt, missing_feedback
from agentic_synth_data.schemas.missing_values import MissingHandling
from agentic_synth_data.schemas.synthesis_request import SynthesisRequest, ImputationConfig


def update_missing_value_choice(request: SynthesisRequest) -> SynthesisRequest:
    """Update missing value handling choice in the case that the remove setting for null values
    is too harsh.

    Args:
        request (SynthesisRequest): User preferences for data synthesis.

    Returns:
        SynthesisRequest: Updated synthesis request.
    """
    choice = pyip.inputMenu(
        choices=["auto", "impute",],
        prompt="\nPlease select another method to handle missing values:\n",
        numbered=True
    )

    request.missing_values.mode = choice

    if choice == "impute":
        cat_choice = pyip.inputMenu(
            choices=get_args(CategoricalChoiceAuto.__value__), # pylint: disable=no-member
            prompt=(
                "\nHow would you like to handle missing categorical values? Select 'auto' "
                "if you would like a suggestion.\n"
            ),
            numbered=True,
        )

        if cat_choice == "most frequent":
            cat_choice = "most_frequent"

        num_choice = pyip.inputMenu(
            choices=get_args(NumericalChoiceAuto.__value__), # pylint: disable=no-member
            prompt=(
                "\nHow would you like to handle missing numerical values? Select 'auto' if you "
                "would like a suggestion.\n"
            ),
            numbered=True,
        )

        if num_choice == "most frequent":
            num_choice = "most_frequent"

        request.missing_values.impute.method = "SimpleImputer"
        request.missing_values.impute.categorical_strategy = cat_choice
        request.missing_values.impute.numerical_strategy = num_choice

    elif choice == "auto":
        request.missing_values.impute = ImputationConfig()

    return request


def update_missing_value_choice_ui(request: SynthesisRequest) -> SynthesisRequest:
    """Update missing value handling choice in the case that remove all setting for null values is
    too harsh (UI).

    Args:
        request (SynthesisRequest): User preferences for data synthesis.

    Returns:
        SynthesisRequest: Updated synthesis request.
    """
    st.write("Please select another method to handle missing values:")

    choice = st.selectbox(
        label="**Update missing value handling choice**",
        options=["auto", "impute",],
        index=None,
        key="update_missing_value_handling_choice",
    )

    if choice is None:
        st.stop()

    if choice == "impute":
        st.write(
            "How would you like to handle missing categorical values? Select 'auto' if you would "
            "like a suggestion."
        )

        cat_choice = st.selectbox(
            label="**Update missing categorical value handling choice**",
            options=get_args(CategoricalChoiceAuto.__value__), # pylint: disable=no-member
            index=None,
            format_func=format_option,
            key="update_categorical_strategy",
        )

        num_choice = st.selectbox(
            label="**Update missing value handling choice**",
            options=get_args(NumericalChoiceAuto.__value__), # pylint: disable=no-member
            index=None,
            format_func=format_option,
            key="update_numerical_strategy",
        )

        if cat_choice is None or num_choice is None:
            st.stop()

        if cat_choice == "most frequent":
            cat_choice = "most_frequent"

        if num_choice == "most frequent":
            num_choice = "most_frequent"

        request.missing_values.mode = "impute"
        request.missing_values.impute.method = "SimpleImputer"
        request.missing_values.impute.categorical_strategy = cat_choice
        request.missing_values.impute.numerical_strategy = num_choice

    elif choice == "auto":
        request.missing_values.mode = "auto"
        request.missing_values.impute = ImputationConfig()

    return request


def auto_update_missing_plan(
    request: SynthesisRequest, missing_plan: dict[str, Any],
) -> SynthesisRequest:
    """When "auto" is selected in the missing value handling plan, the agent is called and 
    returns a structured dictionary. This structured dictionary is changed to the format of
    the synthesis request.

    Args:
        request (SynthesisRequest): User preferences for data synthesis.
        missing_plan (dict[str, Any]): Agent recommendation for handling missing values.

    Returns:
        SynthesisRequest: Updated synthesis request.
    """
    miss = request.missing_values

    miss.mode = missing_plan["mode"]
    miss.impute.method = missing_plan["imputation_method"]
    miss.impute.categorical_strategy = missing_plan["categorical_strategy"]
    miss.impute.numerical_strategy = missing_plan["numerical_strategy"]

    return request


def save_categorical_fill_ui() -> None:
    """Save the confirmed categorical fill value to Streamlit session state."""
    st.session_state["cat_fill_value"] = st.session_state["cat_fill_input"].strip() or "Unknown"
    st.session_state["cat_fill_confirm"] = True


def handle_missing_values(
    df: pd.DataFrame,
    request: SynthesisRequest,
    stypes: SemanticTypes,
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
) -> tuple[pd.DataFrame, SynthesisRequest] | None:
    """Handle missing values in a dataframe based on user defined preferences.

    Missing value strategies include:
    1. Remove all rows containing null values.
    2. Imputation.
    3. Receive a suggestion from an agent on how to handle null values.

    Args:
        df (pd.DataFrame): The input dataframe.
        request (SynthesisRequest): User preferences for data synthesis.
        stypes (SemanticTypes): Map of column names to semantic types (e.g., categorical,
            ordinal, integer, float).
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to recommend 
            a strategy for missing values. Defaults to None.
        config (RunnableConfig | None, optional): Configuration passed to the agent during
            invocation. Defaults to None.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]] | None: 
        pd.DataFrame: The processed dataframe after missing value handling. 
        SynthesisRequest: The updated request.
        None: Agent planned missing value handling, approved by user.
    """
    if df.isna().sum().sum() == 0:
        print(
            "\nThere are no missing values in the dataset. Continuing to synthetic data generation."
        )

        return df, request

    print(
        f"\n{df.isna().sum().sum()} samples with missing values have been detected in the dataset."
    )

    mode = request.missing_values.mode
    impute = request.missing_values.impute

    # 1. Remove all samples with NA columns.
    if mode == "remove":
        null_row_count = df.isna().any(axis=1).sum()
        print(
            f"\nWarning: {null_row_count} out of {df.shape[0]} rows "
            f"({null_row_count/df.shape[0]:.2%}) will be removed from the dataset."
        )

        confirm = pyip.inputYesNo(prompt="\nConfirm whether this should be done (yes/no/y/n):\n")
        if confirm == "yes":
            df = df.dropna()
            print(
                "\n🠊 All samples with missing values have been removed. There are "
                f"{df.isna().sum().sum()} remaining missing values."
            )

            return df, request

        # If this option is too harsh and the user wants to try another method.
        request = update_missing_value_choice(request)

        return handle_missing_values(df, request, stypes, agent, config)

    # 2. Perform imputation given there are decided categorical and numerical strategies.
    if (
        mode == "impute" and
        impute.categorical_strategy != "auto" and
        impute.numerical_strategy != "auto"
    ):
        # Separate user defined categorical and numerical values for imputation
        grouped = {}
        for col, stype in stypes.items():
            grouped.setdefault(stype, []).append(col)

        cat_cols = grouped.get("Categorical", []) + grouped.get("Ordinal", [])
        num_cols = grouped.get("Integer", []) + grouped.get("Float", [])

        df = df.copy()

        print("\nReview of missing value handling results:")

        # Categorical value imputation
        cat_strategy = request.missing_values.impute.categorical_strategy
        if cat_cols:
            cat_data = df[cat_cols].astype("object").where(df[cat_cols].notna(), np.nan)

            if cat_strategy == "constant":
                print(
                    "\nEnter the fill value that missing values should have. Fill values "
                    "must be strings.\nPress enter if you would like to stay with the "
                    'default ("Unknown"):'
                )

                fill_value = pyip.inputStr(blank=True,)

                if fill_value == "":
                    fill_value = "Unknown"

                imputer = SimpleImputer(strategy=str(cat_strategy), fill_value=fill_value)
                print(f"Missing values filled with '{fill_value}'")

            else:
                imputer = SimpleImputer(strategy=str(cat_strategy))

            df[cat_cols] = pd.DataFrame(
                imputer.fit_transform(cat_data),
                columns=cat_cols,
                index=df.index
            )

            print(
                f"- There are {df[cat_cols].isna().sum().sum()} remaining null values "
                "in the categorical columns."
            )

        # Numerical value imputation
        if num_cols:
            num_strategy = request.missing_values.impute.numerical_strategy
            imputer = SimpleImputer(strategy=str(num_strategy))

            df[num_cols] = pd.DataFrame(
                imputer.fit_transform(df[num_cols]),
                columns=df[num_cols].columns,
                index=df.index,
            )

            print(
                f"- There are {df[num_cols].isna().sum().sum()} remaining null values "
                "in the numerical columns."
            )

        print(f"\n🠊 There are {df.isna().sum().sum()} remaining null values in the dataset.")

        return df, request

    # 3. Use the agent to suggest the missing value handling strategy.
    if (
        mode == "auto" or
        impute.categorical_strategy == "auto" or
        impute.numerical_strategy == "auto"
    ):
        proportion_missing = (df.isna().sum() / df.shape[0]).round(4).to_dict()
        rows_with_missing = int(df.isna().any(axis=1).sum())
        total_rows = df.shape[0]

        request_dict = request.model_dump()

        prompt = missing_value_prompt(
            proportion_missing=proportion_missing,
            rows_with_missing=rows_with_missing,
            total_rows=total_rows,
            request=request_dict,
            stypes=stypes,
        )

        print("\nAgent is coming up with a suggestion for missing value handling...")

        feedback = None
        while True:
            if feedback is None:
                current_prompt = prompt

            else:
                current_prompt = missing_feedback(prompt, feedback)
                print("Waiting for agent response...")

            missing_plan = invoke_agent(
                prompt=current_prompt, model_cls=MissingHandling, agent=agent, config=config,
            )

            updated_plan = {
                "mode": missing_plan.mode,
                "imputation_method": missing_plan.imputation_method,
                "categorical_strategy": missing_plan.categorical_strategy, 
                "numerical_strategy": missing_plan.numerical_strategy,
                "warning": missing_plan.warning,
                "rationale": missing_plan.rationale,
            }

            print(
                "\nSuggested plan:\n"
                f"- Mode: {updated_plan['mode']}"
            )

            if updated_plan["imputation_method"] is not None:
                print(f"- Imputation method: {updated_plan['imputation_method']}")

            if updated_plan["categorical_strategy"] is not None:
                print(
                    f"- Categorical strategy: {format_option(updated_plan['categorical_strategy'])}"
                )

            if updated_plan["numerical_strategy"]:
                print(f"- Numerical strategy: {format_option(updated_plan['numerical_strategy'])}")

            if len(updated_plan["warning"]) > 0:
                print("\nWarning:")
                format_response(f"- Warning: {updated_plan['warning'][0]}")

            print("\nReasoning behind the suggestion:")
            format_response(updated_plan["rationale"])

            confirm = pyip.inputYesNo(prompt="\nWould you like to proceed with the suggestion?")
            if confirm == "yes":
                # Update synthesis request with new specifications
                request = auto_update_missing_plan(request, updated_plan)
                print("\n\nThe missing value handling plan has been updated to the following:")

                display_synthesis_request(request)
                break

            print(
                "\n\nPlease describe how you would want missing values to be handled and what "
                "changes you would make to the suggested plan."
            )

            feedback = input()

        # Update synthesis request with new specifications
        request = auto_update_missing_plan(request, updated_plan)

        return handle_missing_values(
            df=df, request=request, stypes=stypes, agent=agent, config=config,
        )


def handle_missing_values_ui(
    df: pd.DataFrame,
    request: SynthesisRequest,
    stypes: SemanticTypes,
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
) -> tuple[pd.DataFrame, SynthesisRequest] | None:
    """Handle missing values in a dataframe based on user defined preferences (UI).

    Missing value strategies include:
    1. Remove all rows containing null values.
    2. Imputation.
    3. Receive a suggestion from an agent on how to handle null values.

    Args:
        df (pd.DataFrame): The input dataframe.
        request (SynthesisRequest): User preferences for data synthesis.
        stypes (SemanticTypes): Map of column names to semantic types (e.g., categorical,
            ordinal, integer, float).
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to 
            recommend a strategy for missing values. Defaults to None.
        config (RunnableConfig | None, optional): Configuration passed to the agent during
            invocation. Defaults to None.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]] | None: 
        pd.DataFrame: The processed dataframe after missing value handling. 
        SynthesisRequest: The updated request.
        None: Agent planned missing value handling, approved by user.
    """
    if df.isna().sum().sum() == 0:
        st.info(
            "⇨ There are no missing values in the dataset. Continue to synthetic data generation."
        )

        return df, request

    st.info(
        f"⇨ {df.isna().sum().sum()} samples with missing values have been detected in the dataset."
    )

    mode = request.missing_values.mode
    impute = request.missing_values.impute

    # If prompted: Use the agent to suggest the missing value handling strategy.
    if (
        mode == "auto" or
        impute.categorical_strategy == "auto" or
        impute.numerical_strategy == "auto" or
        "missing_plan" in st.session_state
    ):
        proportion_missing = (df.isna().sum() / df.shape[0]).round(4).to_dict()
        rows_with_missing = int(df.isna().any(axis=1).sum())
        total_rows = df.shape[0]

        request_dict = request.model_dump()

        prompt = missing_value_prompt(
            proportion_missing=proportion_missing,
            rows_with_missing=rows_with_missing,
            total_rows=total_rows,
            request=request_dict,
            stypes=stypes,
        )

        if "missing_plan" not in st.session_state:
            with st.spinner("Agent is coming up with a suggestion for missing value handling..."):
                missing_plan = invoke_agent(
                    prompt=prompt, model_cls=MissingHandling, agent=agent, config=config, ui=True,
                )

                st.session_state.missing_plan = {
                    "mode": missing_plan.mode,
                    "imputation_method": missing_plan.imputation_method,
                    "categorical_strategy": missing_plan.categorical_strategy, 
                    "numerical_strategy": missing_plan.numerical_strategy,
                    "warning": missing_plan.warning,
                    "rationale": missing_plan.rationale,
                }

        updated_plan = st.session_state.missing_plan

        info_helper_message(
            message="Agent has finished coming up with suggestion.",
            context=(
                "The agent suggested a missing value handling strategy based on:\n"
                "- The proportion of missing values for each column.\n"
                "- The number of rows with missing values.\n"
                "- Total rows in the dataset.\n"
                "- Your synthesis request.\n"
                "- The semantic data types of each column present in the dataset."
            ),
            key="missing_handling_info",
        )

        st.subheader("Agent recommendation")

        with st.container(border=True):
            st.write(
                "#### Suggested plan:\n"
                f"- **Mode:** {updated_plan['mode']}"
            )

            if updated_plan["imputation_method"] is not None:
                st.write(f"- **Imputation method:** {updated_plan['imputation_method']}")

            if updated_plan["categorical_strategy"] is not None:
                st.write(
                    "    - **Categorical strategy:** "
                    f"{format_option(updated_plan['categorical_strategy'])}"
                )

            if updated_plan["numerical_strategy"]:
                st.write(
                    "    - **Numerical strategy:** "
                    f"{format_option(updated_plan['numerical_strategy'])}"
                )

            if len(updated_plan["warning"]) > 0:
                st.warning(f"**Warning:** {updated_plan['warning'][0]}")

            st.divider()

            st.write(
                "#### Rationale:\n"
                f"{updated_plan["rationale"]}"
            )

        if st.session_state.get("missing_handling_complete", False):
            st.info("⇨ Missing value handling plan has been approved.")

            request = st.session_state.approved_missing_request

        else:
            if "missing_check_version" not in st.session_state:
                st.session_state.missing_check_version = 0

            confirm = st.selectbox(
                label="**Would you like to proceed with the suggestion?**",
                options=["yes", "no"],
                index=None,
                key=f"missing_handling_check_suggestion_{st.session_state.missing_check_version}",
            )

            if confirm == "yes":
                # Update synthesis request with new specifications
                request = auto_update_missing_plan(request, updated_plan)

                st.session_state.approved_missing_request = request
                st.session_state.missing_handling_complete = True

                st.info("⇨ The missing value handing plan has been approved.")
                st.rerun()

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
                        missing_plan = invoke_agent(
                            prompt=current_prompt,
                            model_cls=MissingHandling,
                            agent=agent,
                            config=config,
                            ui=True,
                        )

                    st.session_state.missing_plan = {
                        "mode": missing_plan.mode,
                        "imputation_method": missing_plan.imputation_method,
                        "categorical_strategy": missing_plan.categorical_strategy, 
                        "numerical_strategy": missing_plan.numerical_strategy,
                        "warning": missing_plan.warning,
                        "rationale": missing_plan.rationale,
                    }

                    st.session_state.missing_check_version += 1
                    st.rerun()

                st.stop()

            else:
                st.stop()

    # Refresh preferences after approval
    mode = request.missing_values.mode
    impute = request.missing_values.impute

    # 1. Remove all samples with NA columns.
    if mode == "remove":
        null_row_count = df.isna().any(axis=1).sum()
        st.warning(
            f"**Warning:** {null_row_count} out of {df.shape[0]} rows "
            f"({null_row_count/df.shape[0]:.2%}) will be removed from the dataset."
        )

        confirm = st.selectbox(
            label="**Confirm whether this should be done:**",
            options=["yes", "no"],
            index=None,
        )

        if confirm is None:
            st.stop()

        if confirm == "yes":
            df = df.dropna()
            st.info(
                "⇨ All samples with missing values have been removed. There are "
                f"{df.isna().sum().sum()} remaining missing values."
            )

            return df, request

        # If this option is too harsh and the user wants to try another method.
        request = update_missing_value_choice_ui(request)

        return handle_missing_values_ui(df, request, stypes, agent, config)

    # 2. Perform imputation given there are decided categorical and numerical strategies.
    if (
        mode == "impute" and
        impute.categorical_strategy != "auto" and
        impute.numerical_strategy != "auto"
    ):
        # Separate user defined categorical and numerical values for imputation
        grouped = {}
        for col, stype in stypes.items():
            grouped.setdefault(stype, []).append(col)

        cat_cols = grouped.get("Categorical", []) + grouped.get("Ordinal", [])
        num_cols = grouped.get("Integer", []) + grouped.get("Float", [])

        df = df.copy()

        st.divider()

        st.write("#### Review of missing value handling results:")

        # Categorical value imputation
        cat_strategy = request.missing_values.impute.categorical_strategy
        if cat_cols:
            cat_data = df[cat_cols].astype("object").where(df[cat_cols].notna(), np.nan)

            if cat_strategy == "constant":
                if "cat_fill_value" not in st.session_state:
                    st.write(
                        "\nEnter the fill value that missing values should have. Fill values "
                        "must be strings.\nPress enter if you would like to stay with the "
                        'default ("Unknown"):'
                    )

                    with st.form("categorical_fill_form"):
                        custom_value = st.text_input(
                            label="**Fill value**", placeholder="Default: Unknown",
                        )
                        submitted = st.form_submit_button("Confirm")

                    if not submitted:
                        st.stop()

                    st.session_state["cat_fill_value"] = custom_value or "Unknown"

                fill_value = st.session_state["cat_fill_value"]

                imputer = SimpleImputer(strategy=str(cat_strategy), fill_value=fill_value)

                st.info(f'⇨ Missing values filled with "{fill_value}"')

                st.divider()

            else:
                imputer = SimpleImputer(strategy=str(cat_strategy))

            df[cat_cols] = pd.DataFrame(
                imputer.fit_transform(cat_data),
                columns=cat_cols,
                index=df.index,
            )

            st.write(
                f"- There are {df[cat_cols].isna().sum().sum()} remaining null values"
                " in the categorical columns."
            )

        # Numerical value imputation
        if num_cols:
            num_strategy = request.missing_values.impute.numerical_strategy
            imputer = SimpleImputer(strategy=str(num_strategy))

            df[num_cols] = pd.DataFrame(
                imputer.fit_transform(df[num_cols]),
                columns=df[num_cols].columns,
                index=df.index,
            )

            st.write(
                f"- There are {df[num_cols].isna().sum().sum()} remaining null values in the "
                "numerical columns."
            )

        st.info(f"⇨ There are {df.isna().sum().sum()} remaining null values in the dataset.")

        return df, request

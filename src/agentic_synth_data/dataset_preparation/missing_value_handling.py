from typing import Optional, Any, cast, get_args

from langchain_core.runnables import RunnableConfig
import pandas as pd
import numpy as np
import pyinputplus as pyip
from langgraph.graph.state import CompiledStateGraph
from sklearn.impute import SimpleImputer
import streamlit as st

from agentic_synth_data.utils import CategoricalChoiceAuto, NumericalChoiceAuto, format_option, \
    format_response, info_helper_message, invoke_agent, display_synthesis_request, SemanticTypes
from agentic_synth_data.prompts.missing_value_prompt import missing_value_prompt, missing_feedback
from agentic_synth_data.schemas.missing_model import MissingHandling
from agentic_synth_data.schemas.models import SynthesisRequest, ImputationConfig


# ============================================================================
# Update missing value choice (retroactive)
# ============================================================================

def update_missing_value_choice(synthesis_request: SynthesisRequest) -> SynthesisRequest:
    """Update missing value handling choice in the case that either the remove all or
     threshold setting for null values is too harsh.

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.

    Returns:
        SynthesisRequest: Updated synthesis request.
    """
    choice = pyip.inputMenu(
        choices=["auto", "impute", "threshold"],
        prompt="\nPlease select another method to handle missing values:\n",
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
        cat_choice = pyip.inputMenu(
            choices=get_args(CategoricalChoiceAuto.__value__),
            prompt="\nHow would you like to handle missing categorical values? Select 'auto' " \
                "if you would like a suggestion.\n",
            numbered=True,
        )

        if cat_choice == "most frequent":
            cat_choice = "most_frequent"

        num_choice = pyip.inputMenu(
            choices=get_args(NumericalChoiceAuto.__value__),
            prompt=(
                "\nHow would you like to handle missing numerical values? Select 'auto' if you "
                "would like a suggestion.\n"
            ),
            numbered=True,
        )

        if num_choice == "most frequent":
            num_choice = "most_frequent"

        synthesis_request.missing_values.impute.method = "SimpleImputer"
        synthesis_request.missing_values.threshold = None
        synthesis_request.missing_values.impute.categorical_strategy = cat_choice
        synthesis_request.missing_values.impute.numerical_strategy = num_choice

    elif choice == "auto":
        synthesis_request.missing_values.threshold = None
        synthesis_request.missing_values.impute = ImputationConfig()

    return synthesis_request


def update_missing_value_choice_ui(synthesis_request: SynthesisRequest) -> SynthesisRequest:
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

    if choice == "impute":
        st.write(
            "How would you like to handle missing categorical values? Select 'auto' if you would "
            "like a suggestion."
        )

        cat_choice = st.selectbox(
            label="Update missing value handling choice",
            options=get_args(CategoricalChoiceAuto.__value__),
            index=None,
            format_func=format_option,
            key="update_categorical_strategy",
        )

        num_choice = st.selectbox(
            label="Update missing value handling choice",
            options=get_args(NumericalChoiceAuto.__value__),
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

        synthesis_request.missing_values.mode = "impute"
        synthesis_request.missing_values.impute.method = "SimpleImputer"
        synthesis_request.missing_values.threshold = None
        synthesis_request.missing_values.impute.categorical_strategy = cat_choice
        synthesis_request.missing_values.impute.numerical_strategy = num_choice

    elif choice == "auto":
        synthesis_request.missing_values.mode = "auto"
        synthesis_request.missing_values.threshold = None
        synthesis_request.missing_values.impute = ImputationConfig()

    return synthesis_request


def auto_update_missing_plan(
    synthesis_request: SynthesisRequest, missing_plan: dict[str, Any],
) -> SynthesisRequest:
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


def missing_value_summary(
    train: pd.DataFrame, test: pd.DataFrame, cat_cols: list[str], num_cols: list[str],
) -> pd.DataFrame:
    """Summarize remaining missing values by data type.

    Args:
        train (pd.DataFrame): The training dataframe.
        test (pd.DataFrame): The testing dataframe.
        cat_cols (list[str]): List of categorical features in the dataframes.
        num_cols (list[str]): List of numerical features in the dataframes.

    Returns:
        pd.DataFrame: Dataframe demonstrating remaining missing values for both categorical and
            numerical features in train and test datasets.
    """
    counts = {}

    for name, cols in [("Categorical", cat_cols), ("Numerical", num_cols)]:
        if cols:
            counts[name] = {
                "Train": train[cols].isna().sum().sum(),
                "Test": test[cols].isna().sum().sum(),
            }

    return pd.DataFrame.from_dict(counts)


# ============================================================================
# Missing value detection and handling
# ============================================================================

def handle_missing_values(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synthesis_request: SynthesisRequest,
    semantic_types: SemanticTypes,
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, SynthesisRequest] | None:
    """Handle missing values in a dataframe based on user defined preferences.

    Missing value strategies include:
    1. Remove all rows containing null values.
    2. Remove rows exceeding a specified threshold for null values.
    3. Imputation.
    4. Receive a suggestion from an agent on how to handle null values.

    Args:
        train (pd.DataFrame): The training dataframe.
        test (pd.DataFrame): The testing dataframe.
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        semantic_types (SemanticTypes): Map of column names to semantic types (e.g., categorical,
            ordinal, integer, float).
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to recommend 
            a strategy for missing values. Defaults to None.
        config (RunnableConfig | None, optional): Configuration passed to the agent during
            invocation. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None: 
        pd.DataFrame: The processed (train) dataframe after missing value handling.
        pd.DataFrame:The processed (test) dataframe after missing value handling. 
        SynthesisRequest: The updated synthesis_request.
        None: Agent planned missing value handling, approved by user.
    """
    if train.isna().sum().sum() == 0 and test.isna().sum().sum() == 0:
        print(
            "\nThere are no missing values in either the training or test datasets. Continuing "
            "to synthetic data generation."
        )
        return train, test, synthesis_request

    print(
        "\nInitial missing values detected:\n"
        f"- Training set: {train.isna().sum().sum()} of {train.size} total values "
        f"({train.isna().sum().sum() / train.size:.2%})\n"
        f"- Test set: {test.isna().sum().sum()} of {test.size} total values "
        f"({test.isna().sum().sum() / test.size:.2%})"
    )

    mode = synthesis_request.missing_values.mode
    impute = synthesis_request.missing_values.impute

    # 1. Remove all samples with NA columns.
    if mode == "remove":
        null_row_train = train.isna().any(axis=1).sum()
        null_row_test = test.isna().any(axis=1).sum()

        print(
            f"\nWarning:\n"
            f"- Train set: {null_row_train} out of {train.shape[0]} rows "
            f"({null_row_train/train.shape[0]*100}%) will be removed from the dataset.\n"
            f"- Test set: {null_row_test} out of {test.shape[0]} rows "
            f"({null_row_test/test.shape[0]*100}%) will be removed from the dataset."
        )

        confirm = pyip.inputYesNo("\nConfirm whether this should be done (yes/no/y/n):\n")
        if confirm == "yes":
            prep_train = train.dropna()
            prep_test = test.dropna()

            print(
                "\n🠊 All samples with missing values have been removed."
                f"  - The train set has {prep_train.isna().sum().sum()} remaining missing values.\n"
                f"  - The test set has {prep_test.isna().sum().sum()} remaining missing values."
            )

            return prep_train, prep_test, synthesis_request

        # Update method choice if outright removal is too harsh.
        synthesis_request = update_missing_value_choice(synthesis_request)

        return handle_missing_values(train, test, synthesis_request, semantic_types, agent, config)

    # 2. Remove all samples above a specified threshold of NA columns. 
    if mode == "threshold":# NOTE: Threshold is outdated (removed on another branch)
        df = train # NOTE: Threshold is outdated (removed on another branch)
        threshold_samples = df.isna().sum(axis=1) > synthesis_request.missing_values.threshold
        count = threshold_samples.sum()

        print(
            f"Warning: {count} out of {df.shape[0]} ({count/df.shape[0]*100}%) will be removed " \
            "from the dataset."
        )

        confirm = pyip.inputYesNo("\nConfirm whether this should be done.\n")
        if confirm == "yes":
            df = df[~threshold_samples]
            return df, test, synthesis_request # NOTE: Threshold is outdated (removed on another branch)

        confirm = pyip.inputYesNo("\nWould you like to select a new threshold value?\n")
        if confirm == "yes":
            value = pyip.inputInt("\nEnter a new threshold value.\n", min=1) # TODO: Double check how I have it written
            synthesis_request.missing_values.threshold = value

            return handle_missing_values(df, test, synthesis_request, semantic_types, agent, config) # NOTE: Threshold is outdated (removed on another branch)

        synthesis_request = update_missing_value_choice(synthesis_request)
        # TODO add a limit to the threshold such that it doesn't add more
        #  possible TODO allow for threshold changes (if found to be too harsh but want to stay with a threshold)

        return handle_missing_values(df, test, synthesis_request, semantic_types, agent, config) # NOTE: Threshold is outdated (removed on another branch)

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

        train = train.copy()
        test = test.copy()

        print("\nReview of missing value handling results:")
        # Categorical value imputation
        cat_strategy = synthesis_request.missing_values.impute.categorical_strategy
        if cat_cols:
            train_cat = train[cat_cols].astype("object").where(train[cat_cols].notna(), np.nan)

            if cat_strategy == "constant":
                print(
                    "\nEnter the fill value that missing values should have. Fill values must be "
                    "strings.\nPress enter if you would like to stay with the default ('Unknown'):"
                )
                fill_value = pyip.inputStr()

                if fill_value == "":
                    fill_value = "Unknown"

                cat_imputer = SimpleImputer(strategy=cat_strategy, fill_value=fill_value)
                print(f"Missing values filled with '{fill_value}'")

            else:
                cat_imputer = SimpleImputer(strategy=str(cat_strategy))

            train[cat_cols] = pd.DataFrame(
                cat_imputer.fit_transform(train_cat),
                columns=cat_cols,
                index=train.index,
            )

        # Numerical value imputation
        if num_cols:
            num_strategy = synthesis_request.missing_values.impute.numerical_strategy
            num_imputer = SimpleImputer(strategy=str(num_strategy))

            train[num_cols] = pd.DataFrame(
                num_imputer.fit_transform(train[num_cols]),
                columns=num_cols,
                index=train.index,
            )

        missing_df = missing_value_summary(train=train, test=test, cat_cols=cat_cols, num_cols=num_cols,)

        if not missing_df.empty:
            print(
                f"Remaining missing values\n{missing_df.to_string()}\n"
                "Note: Imputation was performed on the training dataset only"
            )

        print(
            f"\n🠊 There are {train.isna().sum().sum() + test.isna().sum().sum()} remaining missing "
            f"values (Train: {train.isna().sum().sum()}, Test: {test.isna().sum().sum()})."
        )

        return train, test, synthesis_request

    # 4. Use the agent to suggest the missing value handling strategy.
    if (
        mode == "auto" or
        impute.categorical_strategy == "auto" or
        impute.numerical_strategy == "auto"
    ):
        # Proportion of missing values for each feature (train and test)
        proportion_missing_train = (train.isna().sum() / train.shape[0]).round(4).to_dict()
        proportion_missing_test = (test.isna().sum() / test.shape[0]).round(4).to_dict()

        # Count of rows with missing values (train and test)
        rows_with_missing_train = int(train.isna().any(axis=1).sum())
        rows_with_missing_test = int(test.isna().any(axis=1).sum())

        # Total rows (train and test)
        total_rows_train = train.shape[0]
        total_rows_test = test.shape[0]

        request_dict = synthesis_request.model_dump()

        prompt = missing_value_prompt(
            proportion_missing_train=proportion_missing_train,
            proportion_missing_test=proportion_missing_test,
            rows_with_missing_train=rows_with_missing_train,
            rows_with_missing_test=rows_with_missing_test,
            total_rows_train=total_rows_train,
            total_rows_test=total_rows_test,
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

            missing_plan = invoke_agent(
                prompt=current_prompt, model_cls=MissingHandling, agent=agent, config=config,
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

            print(
                "\nSuggested plan:\n"
                f"- Mode: {updated_plan['mode']}"
            )

            if updated_plan["set_threshold"] is not None:
                print(f"- Threshold: {updated_plan['set_threshold']}")

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

            confirm = pyip.inputYesNo("\nWould you like to proceed with the suggestion?")
            if confirm == "yes":
                # Update synthesis request with new specifications
                synthesis_request = auto_update_missing_plan(synthesis_request, updated_plan)
                print("\n\nThe missing value handling plan has been updated to the following:")
                display_synthesis_request(synthesis_request)
                break

            print(
                "\n\nPlease describe how you would want missing values to be handled and what "
                "changes you would make to the suggested plan."
            )

            feedback = input()

        # Update synthesis request with new specifications
        synthesis_request = auto_update_missing_plan(synthesis_request, updated_plan)
        return handle_missing_values(train, test, synthesis_request, semantic_types, agent, config)


def handle_missing_values_ui(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synthesis_request: SynthesisRequest,
    semantic_types: SemanticTypes,
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, SynthesisRequest] | None:
    """Handle missing values in a dataframe based on user defined preferences (UI).

    Missing value strategies include:
    1. Remove all rows containing null values.
    2. Remove rows exceeding a specified threshold for null values.
    3. Imputation.
    4. Receive a suggestion from an agent on how to handle null values.

    Args:
        train (pd.DataFrame): The training dataframe.
        test (pd.DataFrame): The testing dataframe.
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
        semantic_types (SemanticTypes): Map of column names to semantic types (e.g., categorical,
            ordinal, integer, float).
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to 
            recommend a strategy for missing values. Defaults to None.
        config (RunnableConfig | None, optional): Configuration passed to the agent during
            invocation. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None: 
        pd.DataFrame: The processed (train) dataframe after missing value handling.
        pd.DataFrame: The processed (test) dataframe after missing value handling. 
        SynthesisRequest: The updated synthesis_request.
        None: Agent planned missing value handling, approved by user.
    """
    if train.isna().sum().sum() == 0 and test.isna().sum().sum() == 0:
        st.write(
            "There are no missing values in either the training or test datasets. Continuing "
            "to synthetic data generation."
        )
        return train, test, synthesis_request

    st.write(
        "#### Initial missing values detected:\n"
        f"- **Training set:** {train.isna().sum().sum()} of {train.size} total values "
        f"({train.isna().sum().sum() / train.size:.2%})\n"
        f"- **Test set:** {test.isna().sum().sum()} of {test.size} total values "
        f"({test.isna().sum().sum() / test.size:.2%})"
    )

    mode = synthesis_request.missing_values.mode
    impute = synthesis_request.missing_values.impute

    # If prompted: Use the agent to suggest the missing value handling strategy.
    if (
        mode == "auto" or
        impute.categorical_strategy == "auto" or
        impute.numerical_strategy == "auto" or
        "missing_plan" in st.session_state
    ):
        # Proportion of missing values for each feature (train and test)
        proportion_missing_train = (train.isna().sum() / train.shape[0]).round(4).to_dict()
        proportion_missing_test = (test.isna().sum() / test.shape[0]).round(4).to_dict()

        # Count of rows with missing values (train and test)
        rows_with_missing_train = int(train.isna().any(axis=1).sum())
        rows_with_missing_test = int(test.isna().any(axis=1).sum())

        # Total rows (train and test)
        total_rows_train = train.shape[0]
        total_rows_test = test.shape[0]

        request_dict = synthesis_request.model_dump()

        prompt = missing_value_prompt(
            proportion_missing_train=proportion_missing_train,
            proportion_missing_test=proportion_missing_test,
            rows_with_missing_train=rows_with_missing_train,
            rows_with_missing_test=rows_with_missing_test,
            total_rows_train=total_rows_train,
            total_rows_test=total_rows_test,
            synthesis_request=request_dict,
            semantic_types=semantic_types,
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
                    "set_threshold": missing_plan.set_threshold,
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

        st.write(
            "#### Suggested plan:\n"
            f"- **Mode:** {updated_plan['mode']}"
        )

        if updated_plan["set_threshold"] is not None:
            st.write(f"- **Threshold:** {updated_plan['set_threshold']}")

        if updated_plan["imputation_method"] is not None:
            st.write(f"- **Imputation method:** {updated_plan['imputation_method']}")

        if updated_plan["categorical_strategy"] is not None:
            st.write(
                f"- **Categorical strategy:** {format_option(updated_plan['categorical_strategy'])}"
            )

        if updated_plan["numerical_strategy"]:
            st.write(
                f"- **Numerical strategy:** {format_option(updated_plan['numerical_strategy'])}"
            )

        if len(updated_plan["warning"]) > 0:
            st.warning(f"- **Warning:** {updated_plan['warning'][0]}")

        st.write("#### Rationale behind the suggestion:")
        st.write(updated_plan["rationale"])

        if st.session_state.get("missing_handling_complete", False):
            st.info("⇨ Missing value handling plan has been approved.")
            synthesis_request = st.session_state.approved_missing_request

        else:
            if "missing_check_version" not in st.session_state:
                st.session_state.missing_check_version = 0

            confirm = st.selectbox(
                label="Would you like to proceed with the suggestion?",
                options=["yes", "no"],
                index=None,
                key=f"missing_handling_check_suggestion_{st.session_state.missing_check_version}",
            )

            if confirm == "yes":
                # Update synthesis request with new specifications
                synthesis_request = auto_update_missing_plan(synthesis_request, updated_plan)

                st.session_state.approved_missing_request = synthesis_request
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
                        "set_threshold": missing_plan.set_threshold,
                        "warning": missing_plan.warning,
                        "rationale": missing_plan.rationale,
                    }

                    st.session_state.missing_check_version += 1
                    st.rerun()

                st.stop()

            else:
                st.stop()

    # Refresh preferences after approval
    mode = synthesis_request.missing_values.mode
    impute = synthesis_request.missing_values.impute

    # 1. Remove all samples with NA columns.
    if mode == "remove":
        null_row_train = train.isna().any(axis=1).sum()
        null_row_test = test.isna().any(axis=1).sum()

        st.warning(
            f"**Warning:**\n"
            f"- **Train set:** {null_row_train} out of {train.shape[0]} rows "
            f"({null_row_train/train.shape[0]*100}%) will be removed from the dataset.\n"
            f"- **Test set:** {null_row_test} out of {test.shape[0]} rows "
            f"({null_row_test/test.shape[0]*100}%) will be removed from the dataset."
        )

        confirm = st.selectbox(
            label="Confirm whether this should be done:",
            options=["yes", "no"],
            index=None,
        )

        if confirm is None:
            st.stop()

        if confirm == "yes":
            prep_train = train.dropna()
            prep_test = test.dropna()

            st.info(
                "\n⇨ All samples with missing values have been removed."
                f"  - The train set has {prep_train.isna().sum().sum()} remaining missing values."
                f"  - The test set has {prep_test.isna().sum().sum()} remaining missing values."
            )
            return prep_train, prep_test, synthesis_request

        # Update method choice if outright removal is too harsh.
        synthesis_request = update_missing_value_choice_ui(synthesis_request)

        return handle_missing_values_ui(
            train, test, synthesis_request, semantic_types, agent, config,
        )

    # 2. Perform imputation given there are decided categorical and numerical strategies.
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

        st.divider()

        st.write("#### Review of missing value handling results:")

        # Categorical value imputation
        cat_strategy = synthesis_request.missing_values.impute.categorical_strategy
        if cat_cols:
            train_cat = train[cat_cols].astype("object").where(train[cat_cols].notna(), np.nan)

            if cat_strategy == "constant":
                st.write(
                    "\nEnter the fill value that missing values should have. Fill values "
                    "must be strings.\nPress enter if you would like to stay with the "
                    "default ('Unknown'):"
                )
                fill_value = st.text_input("fill_value")

                if fill_value == "":
                    fill_value = "Unknown"

                cat_imputer = SimpleImputer(strategy=cat_strategy, fill_value=fill_value)
                st.write(f"Missing values filled with '{fill_value}'")

            else:
                cat_imputer = SimpleImputer(strategy=str(cat_strategy))

            train[cat_cols] = pd.DataFrame(
                cat_imputer.fit_transform(train_cat),
                columns=cat_cols,
                index=train.index,
            )

        # Numerical value imputation
        if num_cols:
            num_strategy = synthesis_request.missing_values.impute.numerical_strategy
            num_imputer = SimpleImputer(strategy=str(num_strategy))

            train[num_cols] = pd.DataFrame(
                num_imputer.fit_transform(train[num_cols]),
                columns=num_cols,
                index=train.index
            )

        missing_df = missing_value_summary(train, test, cat_cols, num_cols)

        if not missing_df.empty:
            st.write("##### Remaining missing values:")
            st.table(missing_df)
            st.caption("**Note:** Imputation was performed on the training dataset only")

        st.info(
            f"⇨ There are {train.isna().sum().sum() + test.isna().sum().sum()} remaining missing "
            f"values (Train: {train.isna().sum().sum()}, Test: {test.isna().sum().sum()})."
        )

        return train, test, synthesis_request

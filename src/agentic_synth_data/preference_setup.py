from typing import Literal, get_args

import pyinputplus as pyip
import streamlit as st

from agentic_synth_data.schemas.models import SynthesisRequest
from agentic_synth_data.utils import GeneratorNameAuto, display_synthesis_request, \
    display_synthesis_request_st, format_option, format_response, output_default_parameters, \
    CategoricalChoiceAuto, NumericalChoiceAuto


def build_synthesis_request(
    sample_size: Literal["fixed_value", "same_as_original"],
    fixed_value: int | None,
    privacy: Literal["standard", "differential_privacy"],
    generator: GeneratorNameAuto,
    missing_values: Literal["auto", "impute", "threshold", "remove"],
    impute_method: Literal["SimpleImputer"] | None=None,
    categorical_strategy: CategoricalChoiceAuto | None=None,
    numerical_strategy: NumericalChoiceAuto | None=None,
) -> SynthesisRequest:
    """Build the starting synthesis request from user configuration.

    Args:
        sample_size (Literal["fixed_value", "same_as_original"]): Sample size configuration mode.
        fixed_value (int): Fixed sample size when using `fixed_value` mode.
        privacy (Literal["standard", "differential_privacy"]): Privacy configuration mode.
        generator (GeneratorNameAuto): Synthetic data generator to use.
        missing_values (Literal["auto", "impute", "threshold", "remove"]): Missing value handling 
            mode.
        impute_method (Literal["SimpleImputer"] | None, optional): Imputation method. Defaults to 
            None.
        categorical_strategy (CategoricalChoiceAuto] | None, optional): Strategy for categorical
            imputation. Defaults to None.
        numerical_strategy (NumericalChoiceAuto | None, optional): Strategy for numerical
            imputation. Defaults to None.

    Returns:
        SynthesisRequest: Configured data synthesis request.
    """
    request = SynthesisRequest()

    # Sample size
    if sample_size == "fixed value":
        request.sample_size.mode = "fixed_value"
        request.sample_size.fixed_value = fixed_value
    else:
        request.sample_size.mode = "same_as_original"

    # Privacy
    if privacy == "yes":
        request.privacy.mode = "differential_privacy"

    # Generator
    request.generator.name = generator
    request.generator.parameters = output_default_parameters(generator)

    # Missing values
    request.missing_values.mode = missing_values

    if missing_values == "impute":
        request.missing_values.impute.method = impute_method

        if categorical_strategy == "most frequent":
            categorical_strategy = "most_frequent"

        if numerical_strategy == "most frequent":
            numerical_strategy = "most_frequent"

        request.missing_values.impute.categorical_strategy = categorical_strategy
        request.missing_values.impute.numerical_strategy = numerical_strategy
    return request


def select_preferences() -> SynthesisRequest:
    """Initial setup of user preferences.

    Returns:
        SynthesisRequest: Configured data synthesis request.
    """
    request = SynthesisRequest()

    fixed_value = None
    impute_method = None
    categorical_strategy = None
    numerical_strategy = None

    # Sample size
    format_response(
        "Would you like to generate the same amount of samples as the dataset or a fixed number" \
        " of samples?"
    )
    sample_size = pyip.inputMenu(["fixed value", "same as original"], numbered=True)

    if sample_size == "fixed value":
        print()
        format_response("How many synthetic samples would you like to generate?")
        fixed_value = pyip.inputInt(min=1)

    # Privacy
    privacy = pyip.inputYesNo("\nAre you interested in differential privacy?\n")

    # Generator
    print()
    format_response(
        "Is there a specific data generator that you want to use on your data? Select " \
        "'auto' if you would like a suggestion."
    )

    generator = pyip.inputMenu(get_args(GeneratorNameAuto.__value__), numbered=True,)

    if generator == "auto":
        print("\t⇨ You will receive a recommendation based on the dataset's characteristics.")
        request.generator.name = generator

    else:
        print(f"\t⇨ Your data will be generated using {generator}.")

    # Missing values
    print()
    format_response(
        "How would you like missing values to be handled? Select 'auto' if you would like a" \
        " suggestion."
    )
    missing_values = pyip.inputMenu(["auto", "impute", "remove"], numbered=True)
    if missing_values == "auto":
        print("\t⇨ You will receive a recommendation based on the dataset's characteristics.")

    elif missing_values == "impute":
        print()
        format_response(
            "Currently, only the SimpleImputer imputation method is currently supported."
        )
        impute_method = "SimpleImputer"
        print("\t⇨ Your missing data will be imputed by SimpleImputer.")

        # Categorical handling
        print()
        format_response(
            "How would you like to handle missing categorical values? Select 'auto' if you " \
            "would like a suggestion."
        )

        categorical_strategy = pyip.inputMenu(
            get_args(CategoricalChoiceAuto.__value__), numbered=True,
        )

        if categorical_strategy == "auto":
            print("\t⇨ You will receive a recommendation based on the dataset's characteristics.")

        else:
            print(
                f"\t⇨ Your missing categorical data will be imputed using {categorical_strategy}."
            )

        # Numerical handling
        print()
        format_response(
            "How would you like to handle missing numerical values? Select 'auto' if you " \
            "would like a suggestion."
        )

        numerical_strategy = pyip.inputMenu(
            get_args(NumericalChoiceAuto.__value__), numbered=True,
        )

        if numerical_strategy == "auto":
            print("\t⇨ You will receive a recommendation based on the dataset's characteristics.")

        else:
            print(f"\t⇨ Your missing numerical data will be imputed using {numerical_strategy}.")

    request = build_synthesis_request(
        sample_size=sample_size,
        fixed_value=fixed_value,
        privacy=privacy,
        generator=generator,
        missing_values=missing_values,
        impute_method=impute_method,
        categorical_strategy=categorical_strategy,
        numerical_strategy=numerical_strategy,
    )
    print()
    print("-"*25+"Your preferences have been recorded"+"-"*25)
    display_synthesis_request(request)
    return request


def select_preferences_ui() -> SynthesisRequest:
    """Initial setup of user preferences (UI).
    
    Returns:
        SynthesisRequest: Configured data synthesis request.
    """
    fixed_value = None
    impute_method = None
    categorical_strategy = None
    numerical_strategy = None

    # Sample size
    st.subheader("How many synthetic samples would you like to generate?")

    st.write(
        "Choose whether to generate the same number of records as your original dataset or specify "
        "a different number (fixed value). The number of synthetic samples does not need to match "
        "the original dataset."
    )

    sample_size = st.selectbox(
        label="sample size",
        options=["fixed value", "same as original"],
        label_visibility="hidden",
        index=None,
        key="preferences_sample_size",
    )

    if sample_size == "fixed value":
        st.write("#### How many synthetic samples would you like to generate?")
        fixed_value = st.number_input("", min_value=1)

        st.info(f"⇨ {fixed_value} synthetic samples will be generated.") # small delay

    # if sample_size == "same as original": # TODO include a method to display sample size
    #     st.info(f"⇨ {fixed_value} synthetic samples will be generated.") # small delay

    st.write("---")

    # Privacy
    st.subheader("Is privacy preservation important in your use case?")

    st.write(
        "Your selection determines how much the LLM agent prioritizes privacy when recommending "
        "and evaluating synthetic data generators, relative to data fidelity and utility. Privacy "
        "risks are evaluated regardless of your selection."
    )

    privacy = st.selectbox(
        label="privacy specification",
        options=["yes", "no"],
        label_visibility="hidden",
        index=None,
        key="preferences_privacy",
    )

    st.write("---")

    # Generator
    st.subheader("Is there a specific data generator that you want to use on your data?")
    st.write(
        "Select 'auto' if you would like a suggestion from the LLM agent. The agent will take in "
        "characteristics of your dataset to inform its suggestion."
    )

    generator = st.selectbox(
        label="data generator",
        options=get_args(GeneratorNameAuto.__value__),
        label_visibility="hidden",
        index=None,
        key="preferences_generator_choice",
    )

    if generator == "auto":
        st.info("⇨ You will receive a recommendation from the agent.")

    else:
        if generator is not None:
            st.info(f"⇨ Your data will be generated using {generator}.")

    st.write("---")

    # Missing values
    st.subheader("How would you like missing values to be handled?")
    st.write(
        "Some synthetic data generators cannot handle missing values and require them to be "
        "removed or imputed before generation. Select 'auto' to receive a recommendation from "
        "the LLM agent based on your dataset."
    )

    missing_values = st.selectbox(
        label="missing strategy",
        options=["auto", "impute", "remove"],
        label_visibility="hidden",
        index=None,
        key="preferences_missing_value_handling",
    )

    if missing_values == "auto":
        st.info("⇨ You will receive a recommendation based on the dataset's characteristics.")

    elif missing_values == "impute":
        st.write(
            "Currently, only the SimpleImputer imputation method is currently supported."
        )
        impute_method = "SimpleImputer"
        st.info("⇨ Your missing data will be imputed by SimpleImputer.")

        # Categorical handling
        st.write(
            "#### How would you like to handle missing categorical values?\n"
            "Select 'auto' if you would like a suggestion from the agent."
        )

        categorical_strategy = st.selectbox(
            label="categorical strategy",
            options=get_args(CategoricalChoiceAuto.__value__),
            format_func=format_option,
            index=None,
            key="preferences_categorical_strategy",
            label_visibility="hidden",
        )

        if categorical_strategy is None:
            st.stop()

        if categorical_strategy == "auto":
            st.info("⇨ You will receive a recommendation based on the dataset's characteristics.")

        elif categorical_strategy is not None:
            st.info(
                "⇨ Your missing categorical data will be imputed using "
                f"{format_option(categorical_strategy)}."
            )

        # Numerical handling
        st.write(
            "#### How would you like to handle missing numerical values?\n"
            "Select 'auto' if you would like a suggestion from the agent."
        )

        numerical_strategy = st.selectbox(
            label="numerical strategy",
            options=get_args(NumericalChoiceAuto.__value__),
            index=None,
            format_func=format_option,
            key="preferences_numerical_strategy",
            label_visibility="hidden",
        )

        if numerical_strategy is None:
            st.stop()

        if numerical_strategy == "auto":
            st.info(
                "⇨ You will receive a recommendation based on the dataset's characteristics."
            )

        elif numerical_strategy is not None:
            st.info(
                "⇨ Your missing numerical data will be imputed using "
                f"{format_option(numerical_strategy)}."
            )

    st.write("---")

    complete = (
        sample_size is not None
        and privacy is not None
        and generator is not None
        and missing_values is not None
    )

    if missing_values == "impute":
        complete = (
            complete and
            categorical_strategy is not None and
            numerical_strategy is not None
        )

    if not complete:
        st.info("Please complete all preference selections.")
        st.stop()

    request = build_synthesis_request(
        sample_size=sample_size,
        fixed_value=fixed_value,
        privacy=privacy,
        generator=generator,
        missing_values=missing_values,
        impute_method=impute_method,
        categorical_strategy=categorical_strategy,
        numerical_strategy=numerical_strategy,
    )

    st.header("Your preferences have been recorded")
    display_synthesis_request_st(request)
    return request

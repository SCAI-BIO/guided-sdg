from typing import Literal

from pydantic import BaseModel, Field
import pyinputplus as pyip
import streamlit as st

from agentic_synth_data.utils import display_synthesis_request, display_synthesis_request_st, format_response, output_default_parameters


class SampleSizeConfig(BaseModel):
    mode: Literal["fixed_value", "same_as_original"] | None = None
    fixed_value: int | None = None


class PrivacyConfig(BaseModel):
    mode: Literal["standard", "differential_privacy"] = "standard"
    sensitive_columns: list[str] = Field(default_factory=list)


class GeneratorConfig(BaseModel):
    name: Literal["auto", "CTGAN", "PATEGAN", "BayesianNet", "PrivBayes"] | None = None
    parameters: str | None = None


class ImputationConfig(BaseModel):
    method: Literal["SimpleImputer"] | None = None
    categorical_strategy: Literal["auto", "most_frequent", "constant"] | None = None
    numerical_strategy: Literal["auto", "mean", "median", "most_frequent"] | None = None


class MissingValueConfig(BaseModel):
    mode: Literal["auto", "impute", "threshold", "remove"] | None = None
    impute: ImputationConfig = Field(default_factory=ImputationConfig)
    threshold: int | None = None


class SynthesisRequest(BaseModel):
    sample_size: SampleSizeConfig = Field(default_factory=SampleSizeConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    missing_values: MissingValueConfig = Field(default_factory=MissingValueConfig)


def build_synthesis_request(
    sample_size,
    fixed_value,
    privacy,
    generator,
    missing_values,
    impute_method=None,
    categorical_strategy=None,
    numerical_strategy=None,
) -> SynthesisRequest:
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
        dict[str, Any]: Dictionary of the user's preferences.
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
    generator = pyip.inputMenu(
        ["auto", "CTGAN", "PATEGAN", "BayesianNet", "PrivBayes"], numbered=True
    )
    if generator == "auto":
        print("\t=> You will receive a recommendation based on the dataset's characteristics.")
        request.generator.name = generator

    else:
        print(f"\t=> Your data will be generated using {generator}.")

    # Missing values
    print()
    format_response(
        "How would you like missing values to be handled? Select 'auto' if you would like a" \
        " suggestion."
    )
    missing_values = pyip.inputMenu(["auto", "impute", "remove"], numbered=True)
    if missing_values == "auto":
        print("\t=> You will receive a recommendation based on the dataset's characteristics.")

    elif missing_values == "impute":
        print()
        format_response(
            "Currently, only the SimpleImputer imputation method is currently supported."
        )
        impute_method = "SimpleImputer"
        print("\t=> Your missing data will be imputed by SimpleImputer.")

        # Categorical handling
        print()
        format_response(
            "How would you like to handle missing categorical values? Select 'auto' if you " \
            "would like a suggestion."
        )
        categorical_strategy = pyip.inputMenu(["auto", "most frequent", "constant",], numbered=True)

        if categorical_strategy == "auto":
            print("\t=> You will receive a recommendation based on the dataset's characteristics.")

        else:
            print(
                f"\t=> Your missing categorical data will be imputed using {categorical_strategy}."
            )

        # Numerical handling
        print()
        format_response(
            "How would you like to handle missing numerical values? Select 'auto' if you " \
            "would like a suggestion."
        )
        numerical_strategy = pyip.inputMenu(
            ["auto", "mean", "median", "most frequent"], numbered=True
        )

        if numerical_strategy == "auto":
            print("\t=> You will receive a recommendation based on the dataset's characteristics.")

        else:
            print(f"\t=> Your missing numerical data will be imputed using {numerical_strategy}.")

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
    fixed_value = None
    impute_method = None
    categorical_strategy = None
    numerical_strategy = None

    # Sample size
    st.subheader(
        "Would you like to generate the same amount of samples as the dataset or a fixed "
        "number of samples?"
    )

    sample_size = st.selectbox(
        "sample size", 
        ["fixed value", "same as original"],
        label_visibility="hidden",
        index=None,
        key="preferences_sample_size",
    )

    if sample_size == "fixed value":
        st.write("#### How many synthetic samples would you like to generate?")
        fixed_value = st.number_input("", min_value=1)

    st.write("---")

    # Privacy
    st.subheader("Are you interested in differential privacy?")
    privacy = st.selectbox(
        "privacy specification", 
        ["yes", "no"],
        label_visibility="hidden",
        index=None,
        key="preferences_privacy",
    )

    st.write("---")

    # Generator
    st.subheader("Is there a specific data generator that you want to use on your data?")
    st.write("Select 'auto' if you would like a suggestion.")

    generator = st.selectbox(
        "data generator",
        [
            "auto",
            "PATEGAN",
            "BayesianNet",
            "PrivBayes",
            "CTGAN",
        ],
        label_visibility="hidden",
        index=None,
        key="preferences_generator_choice",
    )

    if generator == "auto":
        st.info(
            "=> You will receive a recommendation based on the dataset's characteristics."
        )

    else:
        if generator is not None:
            st.info(f"=> Your data will be generated using {generator}.")

    st.write("---")

    # Missing values
    st.subheader("How would you like missing values to be handled?")
    st.write("Select 'auto' if you would like a suggestion.")

    missing_values = st.selectbox(
        "missing strategy", 
        ["auto", "impute", "remove"],
        label_visibility="hidden",
        index=None,
        key="preferences_missing_value_handling",
    )

    if missing_values == "auto":
        st.info(
            "=> You will receive a recommendation based on the dataset's characteristics."
        )

    elif missing_values == "impute":
        st.write(
            "Currently, only the SimpleImputer imputation method is currently supported."
        )
        impute_method = "SimpleImputer"
        st.info("\t=> Your missing data will be imputed by SimpleImputer.")

        # Categorical handling
        st.write(
            "#### How would you like to handle missing categorical values? Select 'auto' if you "
            "would like a suggestion."
        )

        categorical_strategy = st.selectbox(
            "categorical strategy",
            ["auto", "most frequent", "constant",],
            label_visibility="hidden",
            index=None,
            key="preferences_categorical_strategy",
        )

        if categorical_strategy is None:
            st.stop()

        if categorical_strategy == "auto":
            st.info(
                "=> You will receive a recommendation based on the dataset's characteristics."
            )

        elif categorical_strategy is not None:
            st.info(
                "=> Your missing categorical data will be imputed using"
                f" {categorical_strategy}."
            )

        # Numerical handling
        st.write("#### How would you like to handle missing numerical values?")
        st.write("Select 'auto' if you"
            " would like a suggestion."
        )

        numerical_strategy = st.selectbox(
            "numerical strategy", 
            ["auto", "mean", "median", "most frequent"],
            label_visibility="hidden",
            index=None,
            key="preferences_numerical_strategy",
        )

        if numerical_strategy is None:
            st.stop()

        if numerical_strategy == "auto":
            st.info(
                "=> You will receive a recommendation based on the dataset's characteristics."
            )

        elif numerical_strategy is not None:
            st.info(
                f"=> Your missing numerical data will be imputed using {numerical_strategy}."
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
            complete
            and categorical_strategy is not None
            and numerical_strategy is not None
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

from typing import Any, Literal

from pydantic import BaseModel, Field
import pyinputplus as pyip

from agentic_synth_data.utils import format_response, display_synthesis_request


class SampleSizeConfig(BaseModel):
    mode: Literal["fixed_value", "same_as_original"] | None = None
    fixed_value: int | None = None


class PrivacyConfig(BaseModel):
    mode: Literal["standard", "differential_privacy"] = "standard"
    sensitive_columns: list[str] = Field(default_factory=list)


class GeneratorConfig(BaseModel):
    name: Literal["auto", "CTGAN", "PATEGAN", "BayesianNet", "PrivBayes"] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict) # Depends on generators used/available (not important for right now)


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


def select_preferences() -> SynthesisRequest:
    """Initial setup of user preferences.

    Returns:
        dict[str, Any]: Dictionary of the user's preferences.
    """
    request = SynthesisRequest()

    # Fill sample size
    format_response("Would you like to generate the same amount of samples as the dataset or a fixed number of samples?")
    choice = pyip.inputMenu(["fixed value", "same as original"], numbered=True)
    
    if choice == "fixed value":
        request.sample_size.mode = "fixed_value"
        print()
        format_response("How many synthetic samples would you like to generate?")
        value = pyip.inputInt(min=1)
        request.sample_size.fixed_value = value
    
    else:
        request.sample_size.mode = "same_as_original"

    response = pyip.inputYesNo("\nAre you interested in differential privacy?\n")
    if response == "yes":
        request.privacy.mode = "differential_privacy"

    # Fill data generator preferences
    print("\n")
    format_response("Is there a specific data generator that you want to use on your data? Select 'auto' if you would like a suggestion.")
    choice = pyip.inputMenu(["auto", "CTGAN", "PATEGAN", "BayesianNet", "PrivBayes"], numbered=True)
    if choice == "auto":
        print("\t=> You will receive a recommendation based on the dataset's characteristics.")
        request.generator.name = choice
    
    else:
        print(f"\t=> Your data will be generated using {choice}.")
    request.generator.name = choice
        
    # Missing value handling preferences
    print()
    format_response("How would you like missing values to be handled? Select 'auto' if you would like a suggestion.")
    choice = pyip.inputMenu(["auto", "impute", "threshold", "remove"], numbered=True)
    if choice == "auto":
        request.missing_values.mode = choice
        print("\t=> You will receive a recommendation based on the dataset's characteristics.")

    elif choice == "threshold":
        # TODO: Add threshold limit
        request.missing_values.mode = choice
        value = pyip.inputInt("\nWhat is the minimum number of missing columns in a sample remove from the data?")
        request.missing_values.threshold = value

    elif choice == "remove":
        request.missing_values.mode = choice

    elif choice == "impute":
        request.missing_values.mode = choice
        print()
        format_response("Currently, only the SimpleImputer imputation method is currently supported.")
        request.missing_values.impute.method = "SimpleImputer"
        print("\t=> Your missing data will be imputed by SimpleImputer.")
        
        # Specify categorical handling.
        print()
        format_response("How would you like to handle missing categorical values? Select 'auto' if you would like a suggestion.")
        choice = pyip.inputMenu(["auto", "most frequent", "constant",], numbered=True)
        
        if choice == "auto":
            print("\t=> You will receive a recommendation based on the dataset's characteristics.")
        
        if choice == "most frequent":
            choice = "most_frequent"
        
        else:
            print(f"\t=> Your missing categorical data will be imputed using {choice}.")

        request.missing_values.impute.categorical_strategy = choice

        # Specify numerical handling.
        print()
        format_response("How would you like to handle missing numerical values? Select 'auto' if you would like a suggestion.")
        choice = pyip.inputMenu(["auto", "mean", "median", "most frequent"], numbered=True)
        
        if choice == "most frequent":
            choice = "most_frequent"
        
        if choice == "auto":
            print("\t=> You will receive a recommendation based on the dataset's characteristics.")
        
        else:
            print(f"\t=> Your missing numerical data will be imputed using {choice}.")
        request.missing_values.impute.numerical_strategy = choice
 
    print()
    print("-"*25+"Your preferences have been recorded"+"-"*25)
    display_synthesis_request(request)
    return request
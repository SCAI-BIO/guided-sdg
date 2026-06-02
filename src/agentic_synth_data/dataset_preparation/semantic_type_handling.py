import pandas as pd
import pyinputplus as pyip
from typing import Any, Literal

from langgraph.graph.state import CompiledStateGraph

from agentic_synth_data.prompts.stype_prompt import semantic_type_message

from pydantic import RootModel

class STypeInference(RootModel[dict[str, Literal["Float", "Integer", "Boolean", "Categorical", "Ordinal",]]]):
    pass

#report = MissingHandling.model_validate_json(response.content)
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
            "unique_values": df[col].value_counts(dropna=False).head().to_dict(),
            "n_unique": df[col].nunique(dropna=False),
        }
    return profiles


def handle_semantic_types(df:pd.DataFrame, agent:CompiledStateGraph, config:dict) -> dict[str, str]:
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
    # Manual semantic type inference based purely on dtype.
    # possible TODO: incorporate this as a manual way to gather semantic types.
        # Something like this (removing the df editing)
    # choice = pyip.inputMenu(["Auto", "Manual"], numbered=True)
    # if choice == "Manual":
    #     semantic_types = {}
    #     for col in df:
    #         if pd.api.types.is_integer_dtype(df[col]):
    #             semantic_types[col] = "Integer"
    #         elif pd.api.types.is_float_dtype(df[col]):
    #             semantic_types[col] = "Float"
    #         elif pd.api.types.is_string_dtype(df[col]):
    #             df[col] = df[col].astype("category")
    #             semantic_types[col] = "Categorical"

    #     print("\nView of the rudimentary semantic types found for each column:")
    #     print("\nNote: Integer, Float, Categorical, and Ordinal are supported types.")
    #     for col in df.columns:
    #         print(f"{col}: {semantic_types[col]}")

    # 1. Make a profile of the semantic types of each column.
    profiles = semantic_type_profile(df)
    prompt = semantic_type_message(profiles)

    # 2. Agent infers the semantic type of columns based on their profile.
    print("\nAgent is analyzing column data types, unique values, and value distributions to infer semantic types...")
    inference = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        },
        config=config,
    )

    plan = inference['messages'][-1].content.split("</think>")[1].replace("\n", "")
    #print(plan) # Sometimes the agent doesn't format responses appropriately
    semantic_types = STypeInference.model_validate_json(plan).root
    #semantic_types = json.loads(plan)

    # TODO:Include more formal checks that each column is accounted for.
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
            print(f"\nInput column name/number to modify (enter 'exit' (not case-sensitive) or 1 to finish)")
            col = pyip.inputMenu(["Exit"] + df.columns.tolist(), numbered=True)

            if col == "Exit":
                break

            print(f"\nSelected the following column: {col}")
            print(f"Current semantic type is '{semantic_types[col]}'. What should this be modified to?")
            choice = pyip.inputMenu(["Integer", "Float", "Boolean", "Categorical", "Ordinal"], numbered=True)

            semantic_types[col] = choice

            print(f"\n=> Updated semantic type for {col}: {semantic_types[col]}")
            print(f"For reference: Semantic types present in each column:")
            for col, stype in semantic_types.items():
                print(f"{col}: {stype}")
            print("-"*40)
    print()
    print("-"*25+"Final data types present in each column"+"-"*25)
    for col, stype in semantic_types.items():
        print(f"{col}: {stype}")
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

    # same columns/order
    synth = synth[real.columns]

    for col in real.columns:
        stype = semantic_types[col]

        if stype in ["Categorical", "Ordinal", "Boolean"]:
            real[col] = real[col].astype("object")
            synth[col] = synth[col].astype("object")

        elif stype == "Integer":
            real[col] = pd.to_numeric(real[col], errors="coerce").astype("int64")
            synth[col] = pd.to_numeric(synth[col], errors="coerce").astype("int64")

        elif stype == "Float":
            real[col] = pd.to_numeric(real[col], errors="coerce").astype("float64")
            synth[col] = pd.to_numeric(synth[col], errors="coerce").astype("float64")

    return real, synth
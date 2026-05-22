import pandas as pd
import pyinputplus as pyip
import json
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from agentic_synth_data.prompts.stype_prompt import semantic_type_message


def semantic_type_profile(df:pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Create a profile for each column in the dataframe containing the following 
    information (basic): 
    - Column names.
    - Corrected dtypes of each column.
    - Number of unique values (not null) in each column.

    This will be used by the agent to infer the semantic types of each column.

    Args:
        df (pd.DataFrame): The input dataframe with corrected dtypes.

    Returns:
        dict[str, dict[str, Any]: Semantic type profile where keys are column names
        and values are dictionaries which hold information specific to the column.
    """
    profiles = {}

    for col in df.columns:
        profiles[col] = {
            "dtype": str(df[col].dtype),
            "n_unique": df[col].nunique(), # .nunique() default is dropna=True
        }
    return profiles


def handle_semantic_types(df:pd.DataFrame, agent:CompiledStateGraph, config:dict) -> dict[str, str]:
    """Determine the semantic type of each column in a dataframe by:
    1. Making a profile for each column (dtype and number of non null unique values).
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

    print("\n\n")
    print("="*25+"STEP: Handle semantic types"+"="*25)
    # 1. Make a profile of the semantic types of each column.
    profiles = semantic_type_profile(df)
    prompt = semantic_type_message(profiles)

    # 2. Agent infers the semantic type of columns based on their profile.
    print("\nAgent is inferring semantic data types based on dtype and number of unique samples...")
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
    print(plan)
    semantic_types = json.loads(plan)

    # TODO:Include more formal checks that each column is accounted for.
    print("\nDetected semantic types for each column:")
    print("Note: Supported types are Integer, Float, Categorical, and Ordinal.")
    print()
    for col, stype in semantic_types.items():
        print(f"{col}: {stype}")

    # 3. User confirmation.
    confirm = pyip.inputYesNo("\nDo any of the columns need correction?")
    # possible TODO: Add a way to select multiple columns as a group and to change their stype (semantic type).
    
    # 4. If corrections are needed, the user can manually update column types.
    if confirm == "yes":
        while True:
            print(f"\nInput column name/number to modify (enter 'exit' (not case-sensitive) or 1 to finish)")
            col = pyip.inputMenu(["Exit"] + df.columns.tolist(), numbered=True)

            if col == "Exit":
                break

            print(f"\nSelected the following column: {col}")
            print(f"Current semantic type is '{semantic_types[col]}'. What should this be modified to?")
            choice = pyip.inputMenu(["Integer", "Float","Categorical", "Ordinal"], numbered=True)

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


# NOTE: Not needed anymore:
# def match_to_semantic_types(df:pd.DataFrame, semantic_types:dict[str, str]) -> pd.DataFrame:
#     """Converts columns in a dataframe to their specified semantic types.

#     Args:
#         df (pd.DataFrame): The input dataframe.
#         semantic_types (dict[str, str]): The input semantic types.

#     Returns:
#         pd.DataFrame: Updated dataframe explicitly matching specified semantic types.
#     """
#     df = df.copy()
#     for col, semantic_type in semantic_types.items():
#         if semantic_type == "Integer":
#             df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64") # Change?

#         elif semantic_type == "Float":
#             df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64") # Change?

#         elif semantic_type == "Categorical":
#             df[col] = df[col].astype("string").astype("category")

#         elif semantic_type == "Ordinal":
#             dtype = CategoricalDtype(categories=df[col].dropna().unique(), ordered=True)
#             df[col] = df[col].astype("string").astype(dtype)
#     return df
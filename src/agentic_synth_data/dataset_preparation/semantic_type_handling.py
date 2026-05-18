import pandas as pd
import pyinputplus as pyip
from pandas.api.types import CategoricalDtype


# Give user the optionality to define semantic data types
def handle_semantic_types(df):
    semantic_types = {}
    for col in df:
        if pd.api.types.is_integer_dtype(df[col]):
            semantic_types[col] = "Integer"
        elif pd.api.types.is_float_dtype(df[col]):
            semantic_types[col] = "Float"
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("category")
            semantic_types[col] = "Categorical"

    print("View of the rudimentary semantic types found for each column:")
    for col in df.columns:
        print(f"{col}: {semantic_types[col]}")

    print("\nSupported semantic types are as follows: Integer, Float, Categorical, and Ordinal.")
    confirm = pyip.inputYesNo("Do any of the columns need additional specification?")
    # TODO: Add a way to select multiple columns and convert them to a specific datatype all together.
    # either all or a subset (maybe add in an agent here)
    if confirm == "yes":
        while True:
            print(f"\nInput column name/number to modify (enter exit (not case-sensitive) or 1 to finish)")
            var = pyip.inputMenu(["Exit"] + df.columns.tolist(), numbered=True)

            if var == "Exit":
                break

            if var not in df.columns:
                print("Invalid column name.")
                continue

            print(f"Selected the following column: {var}")
            choice = pyip.inputMenu(["Integer", "Float","Categorical", "Ordinal"], numbered=True) 
            
            if choice == "Integer":
                df[var] = pd.to_numeric(df[var], errors="coerce").astype("Int64")
                semantic_types[var] = choice
            
            elif choice == "Float":
                df[var] = pd.to_numeric(df[var], errors="coerce").astype("float64")
                semantic_types[var] = choice

            elif choice == "Categorical":
                df[var] = df[var].astype("category")
                semantic_types[var] = choice

            elif choice == "Ordinal":
                dtype = CategoricalDtype(categories=df[var].dropna(), ordered=True)
                df[var] = (df[var].astype(str).astype(dtype))
                semantic_types[var] = choice
            print(f"\nUpdated semantic type for {var}: {semantic_types[var]}")
            print(f"For reference: Semantic types present in each column:")
            for col in df.columns:
                print(f"{col}: {semantic_types[col]}")
            print("-"*40)
    print()
    print("-"*25+"Final data types present in each column"+"-"*25)
    for col in df.columns:
        print(f"{col}: {semantic_types[col]}")
    return df, semantic_types


def match_to_semantic_types(df, semantic_types):
    df = df.copy()
    for col, semantic_type in semantic_types.items():
        if semantic_type == "Integer":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        elif semantic_type == "Float":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        elif semantic_type == "Categorical":
            df[col] = df[col].astype("string").astype("category")

        elif semantic_type == "Ordinal":
            dtype = CategoricalDtype(categories=df[col].dropna().unique(), ordered=True)
            df[col] = df[col].astype("string").astype(dtype)
    return df
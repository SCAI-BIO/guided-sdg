from typing import Hashable

import pandas as pd
import numpy as np
import pyinputplus as pyip


# TODO: Add when df is saved and updated (print statement)
def missing_value_precheck(df:pd.DataFrame) -> tuple[
    pd.DataFrame, bool, dict[Hashable, int], list[Hashable]
    ]:
    """Prior to assessing dtypes, convert any entries that are flagged as
    potential missing values to np.nan with user approval.

    Three cases:
    1. No missing value marker is found.
    2. Only one missing value marker is found.
    3. More than one missing value marker is found.

    Args:
        df (pd.DataFrame): The input dataframe that may or may not have potential
                           missing values.

    Returns:
        tuple[ pd.DataFrame, bool, dict[Hashable, int], list[Hashable] ]:
        - pd.DataFrame: The dataframe where missing value markers have been replaced
                        with np.nan or the unchanged dataframe if no markers were found.
        - bool: Indicator of whether missing value markers where detected.
        - dict[Hashable, int]: Dictionary of missing value markers detected in the 
                               dataframe and their counts.
        - list[Hashable]: Missing value markers that were replaced.
    """
    # TODO: Add more possible missing values that could be detected.
    possible_missing_values = {
            "na", "n/a", "nan", "none", "null", 
            "-", "--", "", "`", "?", "??", ".", "...", 
            -999, None,
            }
    df_norm = df.map(lambda x: x.lower().strip() if isinstance(x, str) else x)

    norm_stack = df_norm.stack() # Flatten the df into a Series to vertically count values.
    detected_markers = norm_stack[norm_stack.isin(possible_missing_values)].value_counts(dropna=False).to_dict()
    replaced_markers = []

    if len(detected_markers) == 0:
        print("\n=> No potential missing value markers have been detected in the dataset.")
        return df, False, detected_markers, replaced_markers

    markers = list(detected_markers)
    # If more than one potential missing value marker is found
    if len(detected_markers) > 1:
        print("\nThe following potential missing value marker(s) were detected:")
        print(detected_markers)

        while True:
            print("\nSelect what you would like to do, given the markers detected:")
            options = ["Continue without replacement", "Replace all"]#, "Replace specific marker(s)"]
            # TODO: Add an option to replace only specific markers out of the detected list.
            # TODO: Add an option to view the column distribution for each marker (for this multiple case)
            choice = pyip.inputMenu(options, numbered=True)
            
            if choice == "Continue without replacement":
                print("\n=> None of the markers detected have been replaced with a null value. The dataset remains unchanged.")
                return df, False, detected_markers, replaced_markers

            elif choice == "Replace all":
                mask = df_norm.isin(markers)
                replaced_count = int(mask.sum().sum())
                df = df.mask(mask, np.nan)
                print(f"\n=> {replaced_count} of the following markers were replaced with null values: {markers}")
                replaced_markers = markers
                return df, True, detected_markers, replaced_markers
            
            elif choice == "Replace specific marker(s)":
                # TODO: Add option for a specific marker or more to be selected and updated.
                pass

    # Only one potential missing value marker was found.
    else:
        print("\nThe following potential missing value marker was detected:")
        print(detected_markers)

        while True:
            options = ["Continue without replacement", "Replace marker"]#, "View column distribution"]
                # TODO: Add an option to view the column distribution for the marker.
            choice = pyip.inputMenu(options, numbered=True)

            if choice == "Continue without replacement":
                print("\n\tNo values were replaced.")
                return df, False, detected_markers, replaced_markers

            elif choice == "Replace marker":
                print("\nReplacing marker...")
                mask = df_norm.isin(markers)
                replaced_count = int(mask.sum().sum())
                df = df.mask(mask, np.nan)
                print(f"\t{replaced_count} of the following marker were replaced with null values: {markers}")
                replaced_markers = markers
                return df, True, detected_markers, replaced_markers
            
    # possible TODO: Incorporate my previous code such that missing value symbols that aren't
        # already listed can be manually added into the dataset.  
    # # Replace unofficial missing values:
    # confirm = pyip.inputYesNo("Are there any missing values denoted by a character/value other than NA (e.g., '?', '`', 'NA')?")
    # if confirm == "yes":
    #     # TODO maybe incorportae a check if the value is a string or int/float so it can be represented.
    #     value = input("Input the value that will replace all equivalent entries in the dataset:")
    #     total = (df == value).sum().sum()

    #     if total == 0:
    #         print(f"\nThere are none of the specified value ({value}) in the dataset.")
    #     else:
    #         print(f"\nThere are {total} total values that will be replaced with NA in the dataset.")
    #         confirm = pyip.inputYesNo(f"Should the following be replaced with NA?: {value}")
    #         if confirm == "yes":
    #             df = df.replace(value, np.nan)
    #             print(f"\nThe following entry has been replaced with NA in the dataset: {value}")
    #         else:
    #             print(f"\nThe following entry has not been replaced with NA in the dataset: {value}")
    #             # possible TODO: Add attempting another input?


## Maybe could be improved so that multiple columns can be assigned to a specific dtype at once?
def infer_dtype(series:pd.Series) -> str:
    """Infer pandas storage dtypes by taking the majority type of each column.

    Args:
        series (pd.Series): Column in the dataframe, to be assigned a dtype.

    Returns:
        str: Majority data type found (excluding null values).
    """
    non_missing = series.dropna()

    if len(non_missing) == 0:
        return "string"
    
    type_counts = series.dropna().apply(lambda x: type(x).__name__).value_counts(normalize=True)
    majority_type = type_counts.idxmax()

    if majority_type == "str":
        return "string"

    elif majority_type == "int":
        return "int64"
    
    elif majority_type == "float":
        numeric_values = pd.to_numeric(non_missing, errors="coerce").dropna()
        
        if len(numeric_values) == 0:
            return "string"
        
        whole_numbers = ((numeric_values % 1) == 0).all()
        if whole_numbers:
            return "int64"
        return "float64"
    
    # TODO: add additional data types datetime
    return "string"


def handle_dtypes(df:pd.DataFrame) -> tuple[pd.DataFrame, dict[Hashable, str]]:
    """Handles the dtypes for the input dataframe in three main stages:
    1. missing_value_precheck: Initial check for missing value markers, which are intended as (and should be converted 
                               to) null values.
    2. infer_dtype: Convert columns to their inferred dtypes by converting them to their majority dtype (excluding null
                    values).
    3. Dtype correction: If necessary, the user can correct the assumed dtypes manually.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        tuple[pd.DataFrame, dict[Hashable, str]]:
        - pd.DataFrame: The updated dataframe after dtype verification
        - dict[Hashable, str]: Dictionary with the dtype of each column in the dataframe.
    """
    # General check for majority int string or float and convert all in the column main dtype
    for col in df.columns:
        found_dtype = infer_dtype(df[col])
        if found_dtype == "string":
            df[col] = df[col].astype("string")
        # Possibly could set errors to default (raise), but would have to try to handle this.
        elif found_dtype == "int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64") # NOTE: Will likely have to change dtype prior to eval
        elif found_dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    
    print(f"\nCurrent (Python) data types present in each column:\n{df.dtypes}")
    print("\nSupported data types are as follows: Integer, Float, and String.")
    print("Examples:\n- Integer: -23, 0, 34, 120")
    print("- Float: -23.2, 0.6, 34.9, 120.14")
    print("- String: 'red 53', 'S', '12.6%', 'third'")
    confirm = pyip.inputYesNo("\nAre all column data types (Python) correct?\n")
    # TODO: Add a way to select multiple columns and convert them to a specific datatype all together.
    if confirm == "no":
        while True:
            print(f"\nInput column name/number to modify (enter exit (not case-sensitive) or 1 to finish)\n")
            var = pyip.inputMenu(["Exit"] + df.columns.tolist(), numbered=True)

            if var == "Exit":
                break

            if var not in df.columns:
                print("Invalid column name.")
                continue

            print(f"Selected the following column: {var}")
            choice = pyip.inputMenu(["Integer", "Float", "String",], numbered=True) 
            
            if choice == "Integer":
                df[var] = pd.to_numeric(df[var], errors="coerce").astype("int64")
            
            elif choice == "Float":
                df[var] = pd.to_numeric(df[var], errors="coerce").astype("float64")

            elif choice == "String":
                df[var] = df[var].astype("string")

            print(f"\nUpdated dtype for {var}: {df[var].dtype}")
            print(f"For reference: Data types present in each column:\n{df.dtypes}")
            print("-"*40)
    print()
    print("-"*25+"Final data types present in each column"+"-"*25)
    print(df.dtypes)
    py_dtypes = df.dtypes.astype(str).to_dict()
    return df, py_dtypes


# def eval_change_dtypes(df):
#     for col in df.columns:
#     found_dtype = infer_dtype(df[col])
#     if found_dtype == "string":
#         df[col] = df[col].astype("string")
#     # Possibly could set errors to default (raise), but would have to try to handle this.
#     elif found_dtype == "int64":
#         df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64") # NOTE: Will likely have to change dtype prior to eval
#     elif found_dtype == "float64":
#         df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
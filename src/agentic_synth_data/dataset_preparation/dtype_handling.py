import pandas as pd
import numpy as np
import pyinputplus as pyip


## Maybe could be improved so that multiple columns can be assigned to a specific dtype at once?
def infer_dtype(series):
    """Infers pandas storage dtypes by taking the majority type of each column"""
    non_missing = series.dropna()

    if len(non_missing) == 0:
        return "string"
    
    type_counts = series.dropna().apply(lambda x: type(x).__name__).value_counts(normalize=True)
    majority_type = type_counts.idxmax()

    if majority_type == "str":
        return "string"

    elif majority_type == "int":
        return "Int64"
    
    elif majority_type == "float":
        whole_numbers = ((non_missing % 1) == 0).all()
        if whole_numbers:
            return "Int64"
        return "float64"
    
    # TODO: add additional data types datetime
    return "string"


# TODO Consider what to do with Datetime values (have to likely test with the tool if it can handle that)

# Check for unofficial missing values which can initially be encoded as NA
# Show data types present in each column.
# Try to match dtypes to the majority present in each column. 
# (not of concern for right now) If there is a larger issues with a column, possibly employ agent to assist with cleaning the data
" "# Show updated data types present in each column (if there was a change)
# Confirm with user that the data types are correct
# if any are incorrect, have user specify the column (possibly one  by one) or might be easier to add an agent if multiple columns are affected


def handle_dtypes(df:pd.DataFrame):
    # Missing values can affect data types
    # TODO: add a missing value heuristic
    # These are just some missing values
    # It would then make sense to normalize the dataset (stripped and all lowercase at least)
    possible_missing_values = [
        "NA", "N/A", "nan", "None", "NULL", 
        "-", "--", "", " ", "`", "?", "??", ".", "...", 
        -999, None
        ]
    # probably add an option for a deeper look (see which columns)
    # TODO: instead of having the user select this, have it automatically done and count the number of instances of occurence.
    # 
    # Replace unofficial missing values:
    confirm = pyip.inputYesNo("Are there any missing values denoted by a character/value other than NA (e.g., '?', '`', 'NA')?")
    if confirm == "yes":
        # TODO maybe incorportae a check if the value is a string or int/float so it can be represented.
        value = input("Input the value that will replace all equivalent entries in the dataset:")
        total = (df == value).sum().sum()

        if total == 0:
            print(f"\nThere are none of the specified value ({value}) in the dataset.")
        else:
            print(f"\nThere are {total} total values that will be replaced with NA in the dataset.")
            confirm = pyip.inputYesNo(f"Should the following be replaced with NA?: {value}")
            if confirm == "yes":
                df = df.replace(value, np.nan)
                print(f"\nThe following entry has been replaced with NA in the dataset: {value}")
            else:
                print(f"\nThe following entry has not been replaced with NA in the dataset: {value}")
                # possible TODO: Add attempting another input?

    # General check for majority int string or float and convert all in the column main dtype
    for col in df.columns:
        found_dtype = infer_dtype(df[col])
        if found_dtype == "string":
            df[col] = df[col].astype("string")
        # Possibly could set errors to default (raise), but would have to try to handle this.
        elif found_dtype == "Int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif found_dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    
    print(f"\nCurrent data types present in each column:\n{df.dtypes}")
    print("\nSupported data types are as follows: Integer, Float, and String.")
    confirm = pyip.inputYesNo("Do any of the columns need a correction?")
    # TODO: Add a way to select multiple columns and convert them to a specific datatype all together.
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
            choice = pyip.inputMenu(["Integer", "Float", "String",], numbered=True) 
            
            if choice == "Integer":
                df[var] = pd.to_numeric(df[var], errors="coerce").astype("Int64")
            
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
    return df

#df = handle_dtypes(df)
# TODO: Add some sort of save (think about how to store for easy retrieval)
# maybe saving and storing the path of each resulting dataframe after each block 
# as a dictionary of lists where msot recent entries are at the end fo the lists (but that way, everything can be easily retrived)
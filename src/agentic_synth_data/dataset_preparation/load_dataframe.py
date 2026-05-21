import pandas as pd

# possible TODO: see about inclusion of multiple files (the previous 
# project had a multiple file uploader, which could be worth looking
# into.

def load_df(data_path:str) -> pd.DataFrame:#, selected_columns:list) -> pd.DataFrame:
    #path = data_path.lower()

    # if path.endswith(".xlsx"):
    # If this shoudl be added, needs some for cols that are not selected
    #     df = pd.read_excel(data_path, usecols=selected_columns)
    #     if df.shape[0] > 1 and df.shape[1] > 1:
    #         raise ValueError("Invalid entry. Verify excel file is in a tidy format (example: https://aeturrell.github.io/python4DS/data-tidy.html).")
    #     return df

    for sep in [",", "\t", ";", "|"]:
        try:
            # NOTE: Commented out the check to load only user selected columns.
                # Now just assuming the user wants to keep every original column.
            #if len(selected_columns) == 0:
            df = pd.read_csv(data_path, delimiter=sep)
            if df.shape[0] > 1 and df.shape[1] > 1:
                return df
            # else:
            #     df = pd.read_csv(data_path, usecols=selected_columns, delimiter=sep)
            #     if df.shape[0] > 1 and df.shape[1] > 1:
            #         return df
            
        except Exception:
            continue

    raise ValueError("Invalid entry; File type may not be supported. Verify file is in a tidy format (example: https://aeturrell.github.io/python4DS/data-tidy.html).")
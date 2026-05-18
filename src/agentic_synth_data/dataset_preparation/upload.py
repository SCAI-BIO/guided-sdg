import pandas as pd


def upload_data(upload_path:str, selected_columns:list) -> pd.DataFrame:
    """
    Verify uploaded data can be made into a pandas dataframe.
    Currently only tested for one file, would need to be updated to
    account for multiple files (synthcraft has a multiple file uploader
    which might be worth looking into)
    """
    #path = upload_path.lower()

    # if path.endswith(".xlsx"):
    # If this shoudl be added, needs some for cols that are not selected
    #     df = pd.read_excel(upload_path, usecols=selected_columns)
    #     if df.shape[0] > 1 and df.shape[1] > 1:
    #         raise ValueError("Invalid entry. Verify excel file is in a tidy format (example: https://aeturrell.github.io/python4DS/data-tidy.html).")
    #     return df

    for sep in [",", "\t", ";", "|"]:
        try:
            if len(selected_columns) == 0:
                df = pd.read_csv(upload_path, delimiter=sep)
                if df.shape[0] > 1 and df.shape[1] > 1:
                    return df
            else:
                df = pd.read_csv(upload_path, usecols=selected_columns, delimiter=sep)
                if df.shape[0] > 1 and df.shape[1] > 1:
                    return df
            
        except Exception:
            continue

    raise ValueError("Invalid entry; File type may not be supported. Verify file is in a tidy format (example: https://aeturrell.github.io/python4DS/data-tidy.html).")
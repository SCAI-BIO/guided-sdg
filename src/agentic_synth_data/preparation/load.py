import pandas as pd


def load_df(data_path: str) -> pd.DataFrame:
    """Load a dataset file into a pandas DataFrame.

    Args:
        data_path (str): Path to the dataset file.

    Raises:
        ValueError: Raised if the file cannot be loaded as a valid dataframe
                    and recommends tidy dataset formatting.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    # TODO: Include additional file types (e.g., .xlsx)
    # possible TODO: Option to load multiple dataset files (more than one file entry).

    for sep in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(data_path, delimiter=sep)
            if df.shape[0] > 1 and df.shape[1] > 1:
                return df

        except Exception:
            continue

    raise ValueError(
        "Invalid entry; File type may not be supported. Verify file is in a tidy format "
        "(example: https://aeturrell.github.io/python4DS/data-tidy.html)."
    )

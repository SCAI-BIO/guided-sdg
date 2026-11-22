import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_missingness(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split the dataset into training and test sets using missingness stratification when feasible.

    Args:
        df (pd.DataFrame): Input dataset to split.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, str]:
        pd.DataFrame: Training dataset.
        pd.DataFrame: Testing dataset.
        str: Splitting method used ("missing_stratified" or "random")
    """
    # Proportion of missing values per row with missing counts
    missing_ratio = df.isna().mean(axis=1)
    counts = missing_ratio.value_counts()

    # Number of observations in each split
    n_test = int(np.ceil(len(df) * 0.2))
    n_train = len(df) - n_test

    # Stratify only if multiple missingness levels exist and each stratum fits in both splits
    can_stratify = (
        len(counts) > 1 and counts.min() >= 2 and n_test >= len(counts) and n_train >= len(counts)    
    )

    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=missing_ratio if can_stratify else None,
    )

    split_method = "missing_stratified" if can_stratify else "random"
    return train, test, split_method

from collections.abc import Hashable

import pandas as pd
import numpy as np
import pyinputplus as pyip
import streamlit as st

from agentic_synth_data.schemas.missing_values import MissingPrecheckReport
from agentic_synth_data.utils import df_formatter_st


def missing_value_precheck(df: pd.DataFrame) -> MissingPrecheckReport:
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
        MissingPrecheckReport: Report indicating whether missing value markers were detected.
        pd.DataFrame: The dataframe where missing value markers have been replaced
            with np.nan or the unchanged dataframe if no markers were found.
        bool: Indicator of whether missing value markers where detected.
        dict[Hashable, int]: Dictionary of missing value markers detected in the 
            dataframe and their counts.
        list[Hashable]: Missing value markers that were replaced.
    """
    df = df.copy()
    # TODO: Add more possible missing values that could be detected.
    possible_missing_values = {
            "na", "n/a", "nan", "none", "null", 
            "-", "--", "", "`", "?", "??", ".", "...", 
            -999, None,
    }

    df_norm = df.map(lambda x: x.lower().strip() if isinstance(x, str) else x)

    norm_stack = df_norm.stack() # Flatten the df into a Series to vertically count values.
    detected_markers = norm_stack[
        norm_stack.isin(possible_missing_values)
    ].value_counts(dropna=False).to_dict()

    replaced_markers = []
    if len(detected_markers) == 0:
        print("\n🠊 No potential missing value markers have been detected in the dataset.")

        return MissingPrecheckReport(
            df=df,
            has_missing_markers=False,
            detected_markers=detected_markers,
            replaced_markers=replaced_markers
        )

    markers = list(detected_markers)
    # If more than one potential missing value marker is found
    if len(detected_markers) > 1:
        print(
            f"\nThe following potential missing value marker(s) were detected: {detected_markers}"
        )

        while True:
            print("\nSelect what you would like to do, given the markers detected:")

            options = [
                "Continue without replacement", 
                "Replace all",
                #"Replace specific marker(s)",
            ]

            # TODO: Add an option to replace only specific markers out of the detected list.
            # TODO: Add an option to view the column distribution for each marker (for this multiple case)
            choice = pyip.inputMenu(choices=options, numbered=True,)

            if choice == "Continue without replacement":
                print(
                    "\n🠊 None of the markers detected have been replaced with a null value. The "
                    "dataset remains unchanged."
                )

                return MissingPrecheckReport(
                    df=df,
                    has_missing_markers=False,
                    detected_markers=detected_markers,
                    replaced_markers=replaced_markers,
                )

            if choice == "Replace all":
                mask = df_norm.isin(markers)
                replaced_count = int(mask.sum().sum())
                df = df.mask(mask, np.nan)

                print(
                    f"\n🠊 {replaced_count} of the following markers were replaced with null "
                    f"values: {markers}"
                )

                replaced_markers = markers
                return MissingPrecheckReport(
                    df=df,
                    has_missing_markers=True,
                    detected_markers=detected_markers,
                    replaced_markers=replaced_markers,
                )

            if choice == "Replace specific marker(s)":
                # TODO: Add option for a specific marker or more to be selected and updated.
                pass

    # Only one potential missing value marker was found.
    else:
        print(
            f"\nThe following potential missing value marker(s) were detected: {detected_markers}"
        )

        while True:
            options = [
                "Continue without replacement",
                "Replace marker",
                #"View column distribution",
            ]

            # TODO: Add an option to view the column distribution for the marker.
            choice = pyip.inputMenu(choices=options, numbered=True,)

            if choice == "Continue without replacement":
                print("\n\tNo values were replaced.")
                return MissingPrecheckReport(
                    df=df,
                    has_missing_markers=False,
                    detected_markers=detected_markers,
                    replaced_markers=replaced_markers,
                )

            if choice == "Replace marker":
                print("\nReplacing marker...")
                mask = df_norm.isin(markers)
                replaced_count = int(mask.sum().sum())
                df = df.mask(mask, np.nan)

                print(
                    f"\t{replaced_count} of the following marker were replaced with null "
                    f"values: {markers}"
                )

                replaced_markers = markers
                return MissingPrecheckReport(
                    df=df,
                    has_missing_markers=True,
                    detected_markers=detected_markers,
                    replaced_markers=replaced_markers,
                )


def missing_value_precheck_ui(df: pd.DataFrame) -> MissingPrecheckReport:
    """Prior to assessing dtypes, convert any entries that are flagged as
    potential missing values to np.nan with user approval (UI).

    Three cases:
    1. No missing value marker is found.
    2. Only one missing value marker is found.
    3. More than one missing value marker is found.

    Args:
        df (pd.DataFrame): The input dataframe that may or may not have potential
                           missing values.

    Returns:
        MissingPrecheckReport: Report indicating whether missing value markers were detected.
        pd.DataFrame: The dataframe where missing value markers have been replaced
            with np.nan or the unchanged dataframe if no markers were found.
        bool: Indicator of whether missing value markers where detected.
        dict[Hashable, int]: Dictionary of missing value markers detected in the 
            dataframe and their counts.
        list[Hashable]: Missing value markers that were replaced.
    """
    st.write(
        "All missing values within the dataset should be encoded uniformly as null values. If "
        "this is not the case, there can be issues during synthetic generation which are due to "
        "mixing data types or inclusion of characters that do not correspond to the expected "
        "groups. This step checks for any character that might be a placeholder for "
        "a missing value in your dataset (e.g., `n/a`, `?`, `...`, `-999`)."
    )

    df = df.copy()

    # TODO: Add more possible missing values that could be detected.
    possible_missing_values = {
            "na", "n/a", "nan", "none", "null", 
            "-", "--", "", "`", "?", "??", ".", "...", 
            -999, None,
    }

    df_norm = df.map(lambda x: x.lower().strip() if isinstance(x, str) else x)

    norm_stack = df_norm.stack() # Flatten the df into a Series to vertically count values.
    detected_markers = norm_stack[
        norm_stack.isin(possible_missing_values)
    ].value_counts(dropna=False).to_dict()

    replaced_markers = []
    if len(detected_markers) == 0:
        st.info("\n⇨ No potential missing value markers have been detected in the dataset.")

        return MissingPrecheckReport(
            df=df,
            has_missing_markers=False,
            detected_markers=detected_markers,
            replaced_markers=replaced_markers,
        )

    markers = list(detected_markers)
    # If more than one potential missing value marker is found
    if len(detected_markers) > 1:
        st.write("The following potential missing value marker(s) were detected:")

        single_marker_table = pd.DataFrame(detected_markers.items(), columns=["Marker", "Count"],)

        df_formatter_st(single_marker_table)

        st.write("\nSelect what you would like to do, given the markers detected:")
        options = [
            "Continue without replacement", 
            "Replace all",
            #"Replace specific marker(s)",
        ]

        # TODO: Add an option to replace only specific markers out of the detected list.
        # TODO: Add an option to view the column distribution for each marker (for this multiple case)
        choice = st.selectbox(
            "select_markers",
            options,
            index=None,
            label_visibility="hidden",
            key="missing-value-marker-decision-multiple",
        )

        if choice == "Continue without replacement":
            st.info(
                "⇨ None of the markers detected have been replaced with a null value. The"
                " dataset remains unchanged."
            )

            return MissingPrecheckReport(
                df=df,
                has_missing_markers=False,
                detected_markers=detected_markers,
                replaced_markers=replaced_markers,
            )

        if choice == "Replace all":
            mask = df_norm.isin(markers)
            replaced_count = int(mask.sum().sum())
            df = df.mask(mask, np.nan)

            st.write(
                f"⇨ {replaced_count} of the following markers were replaced with null"
                f" values: {markers}"
            )

            replaced_markers = markers
            return MissingPrecheckReport(
                df=df,
                has_missing_markers=True,
                detected_markers=detected_markers,
                replaced_markers=replaced_markers,
            )

        if choice == "Replace specific marker(s)":
            # TODO: Add option for a specific marker or more to be selected and updated.
            pass

    # Only one potential missing value marker was found.
    else:
        st.write("The following potential missing value marker was detected:")

        single_marker_table = pd.DataFrame(detected_markers.items(), columns=["Marker", "Count"],)
        df_formatter_st(single_marker_table)

        st.write("How would you like to handle this missing value marker?")

        options = [
            "Continue without replacement",
            "Replace marker",
            #"View column distribution",
        ]

        # TODO: Add an option to view the column distribution for the marker.
        choice = st.selectbox(
            "markers",
            options,
            index=None,
            label_visibility="hidden",
            key="missing-value-marker-decision-one",
        )

        if choice == "Continue without replacement":
            st.info("No values were replaced.")

            return MissingPrecheckReport(
                df=df,
                has_missing_markers=False,
                detected_markers=detected_markers,
                replaced_markers=replaced_markers,
            )

        if choice == "Replace marker":
            st.write("\nReplacing marker...")

            mask = df_norm.isin(markers)
            replaced_count = int(mask.sum().sum())
            df = df.mask(mask, np.nan)

            st.info(
                f"{replaced_count} of the following marker were replaced with null"
                f" values: {markers}"
            )

            replaced_markers = markers
            return MissingPrecheckReport(
                df=df,
                has_missing_markers=True,
                detected_markers=detected_markers,
                replaced_markers=replaced_markers,
            )


def infer_dtype(series: pd.Series) -> str:
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

    if majority_type == "int":
        return "int64"

    if majority_type == "float":
        numeric_values = pd.to_numeric(non_missing, errors="coerce").dropna()

        if len(numeric_values) == 0:
            return "string"

        whole_numbers = ((numeric_values % 1) == 0).all()
        if whole_numbers:
            return "int64"

        return "float64"

    # TODO: add additional data types datetime
    return "string"


def infer_and_convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """For each column in a dataframe, check for majority data type and convert all
    values to the primary dtype.

    Args:
        df (pd.DataFrame): The updated dataframe after screening for potential missing value 
            markers.

    Returns:
        pd.DataFrame: The dataframe with corrected dtypes.
    """
    df = df.copy()

    for col in df.columns:
        found_dtype = infer_dtype(df[col])

        if found_dtype == "string":
            df[col] = df[col].astype("string")

        # Possibly could set errors to default (raise), but would have to try to handle this.
        elif found_dtype == "int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        elif found_dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


def handle_dtypes(df:pd.DataFrame) -> tuple[pd.DataFrame, dict[Hashable, bool]]:
    """Handles the dtypes for the input dataframe in three main stages:
    1. missing_value_precheck: Initial check for missing value markers, which are intended
                               as (and should be converted to) null values.
    2. infer_dtype: Convert columns to their inferred dtypes by converting them to their 
                    majority dtype (excluding null values).
    3. Dtype correction: If necessary, the user can correct the assumed dtypes manually.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        tuple[pd.DataFrame, dict[Hashable, str]]:
        - pd.DataFrame: The updated dataframe after dtype verification
        - dict[Hashable, str]: Dictionary with the dtype of each column in the dataframe.
    """
    df = df.copy()

    df = infer_and_convert_dtypes(df)

    print(f"\nCurrent (Python) data types present in each column:\n{df.dtypes}")
    print("\nSupported data types are as follows: Integer, Float, and String.")
    print("Examples:\n- Integer: -23, 0, 34, 120")
    print("- Float: -23.2, 0.6, 34.9, 120.14")
    print("- String: 'red 53', 'S', '12.6%', 'third'")
    confirm = pyip.inputYesNo(prompt="\nAre all column data types (Python) correct?\n")

    if confirm == "no":
        while True:
            print(
                "\nInput column name/number to modify (enter exit (not case-sensitive) or 1 "
                "to finish)\n"
            )

            var = pyip.inputMenu(choices=["Exit"] + df.columns.tolist(), numbered=True,)

            if var == "Exit":
                break

            if var not in df.columns:
                print("Invalid column name.")
                continue

            print(f"Selected the following column: {var}")
            choice = pyip.inputMenu(choices=["Integer", "Float", "String",], numbered=True,)

            if choice == "Integer":
                df[var] = pd.to_numeric(df[var], errors="coerce").astype("int64")

            elif choice == "Float":
                df[var] = pd.to_numeric(df[var], errors="coerce").astype("float64")

            elif choice == "String":
                df[var] = df[var].astype("string")

            print(
                f"\nUpdated dtype for {var}: {df[var].dtype}\n"
                f"For reference: Data types present in each column:\n{df.dtypes}"
            )

            print("-" * 40)

    print()
    print("-" * 25 + "Final data types present in each column" + "-" * 25)

    print(df.dtypes)

    return df, df.dtypes.astype(str).to_dict()


def handle_dtypes_ui(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[Hashable, bool]]:
    """Handles the dtypes for the input dataframe in three main stages (UI):
    1. missing_value_precheck_ui: Initial check for missing value markers, which are intended
                               as (and should be converted to) null values.
    2. infer_dtype: Convert columns to their inferred dtypes by converting them to their 
                    majority dtype (excluding null values).
    3. Dtype correction: If necessary, the user can correct the assumed dtypes manually.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        tuple[pd.DataFrame, dict[Hashable, str]]:
        pd.DataFrame: The updated dataframe after dtype verification.
        dict[Hashable, str]: Dictionary with the dtype of each column in the dataframe.
    """
    df = df.copy()

    df = infer_and_convert_dtypes(df)

    st.write("**Current (Python) data types present in each column:**")

    dtype_table = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(dtype) for dtype in df.dtypes],
    })

    df_formatter_st(dtype_table)

    confirm = st.selectbox(
        "**Are all column data types (Python) correct?**",
        ["yes", "no"],
        index=None,
        help=(
        "Supported data types are as follows: Integer, Float, and String.\n\n"
        "Examples:\n"
        "- Integer: -23, 0, 34, 120\n"
        "- Float: -23.2, 0.6, 34.9, 120.14\n"
        "- String: 'red 53', 'S', '12.6%', 'third'"
        ),
    )

    if confirm is None:
        st.stop()

    if confirm == "no":
        st.write("Adjust any incorrect column data types below.")

        with st.form("dtype_correction_form"):
            dtype_choices = {}

            for col in df.columns:
                current_dtype = str(df[col].dtype)

                if current_dtype in ["int64", "Int64"]:
                    current_choice = "Integer"
                elif current_dtype == "float64":
                    current_choice = "Float"
                else:
                    current_choice = "String"

                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(col)

                with col2:
                    st.write(f"Current: **{current_dtype}**")

                with col3:
                    dtype_choices[col] = st.selectbox(
                        "New type",
                        ["Integer", "Float", "String"],
                        index=["Integer", "Float", "String"].index(current_choice),
                        key=f"dtype_{col}",
                        label_visibility="collapsed",
                    )

            submitted = st.form_submit_button("Apply changes")

        if not submitted:
            st.stop()

        st.subheader("Final data types present in each column")

        dtype_table = pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(dtype) for dtype in df.dtypes],
        })

        df_formatter_st(dtype_table)

    return df, df.dtypes.astype(str).to_dict()

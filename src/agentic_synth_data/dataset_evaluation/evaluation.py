from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import pprint

from anonymeter.evaluators import InferenceEvaluator, LinkabilityEvaluator, SinglingOutEvaluator
from IPython.display import Image, display
from langgraph.graph.state import CompiledStateGraph
import numpy as np
import pandas as pd
import pyinputplus as pyip
import streamlit as st
import syndat

from agentic_synth_data.schemas.attack_model import AttackMetrics, AttackColumns, AttackOutputs
from agentic_synth_data.schemas.models import AgentEval, SyndatOutputs
from agentic_synth_data.schemas.streamlit_helpers import StreamlitOutput
from agentic_synth_data.utils import df_formatter_st, format_response, extract_output, \
    info_helper_message, parse_json, parse_json_ui
from agentic_synth_data.prompts.evaluation_prompt import attack_column_selection_message, \
    eval_review_message, evaluation_prompt
from agentic_synth_data.preference_setup import SynthesisRequest
from agentic_synth_data.report import Report


anonymeter_metric_descriptions = {
    "singling_out": (
        "Singling out risk, when it is possible to determine a single data record inside the "
        "original dataset."
    ),
    "linkability": (
        "Linkability risk, when it is possible to link two or more records within the same dataset "
        "or in different ones."
    ),
    "inference": (
        "Inference risk, when an attacker can confidently infer the value of an unknown variable "
        "and its value of the original data."
    ),
}


@dataclass
class AnonymeterInput:
    train_df:pd.DataFrame
    test_df:pd.DataFrame
    synthetic_df:pd.DataFrame
    missing_markers:dict[str, str | int | float]


def calculate_metrics(real_df:pd.DataFrame, synthetic_df:pd.DataFrame) -> dict[str, Any]:
    """Calculate metrics comparing the original dataframe (optionally, with missing value markers
     removed) with the synthetic dataframe.

    Calculates SYNDAT metrics:
    - Jensen-Shannon Divergence
    - Normalized Correlation Difference
    - Discriminator AUC

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synthetic_df (pd.DataFrame): The synthetically generated dataframe.

    Returns:
        dict: The decision plan based on the agent's assessment.
    """
    # TODO: Add an output measuring variant for CLI functions
    jsd_results = syndat.metrics.jensen_shannon_distance(real=real_df, synthetic=synthetic_df)
    jsd_results["jsd_AVERAGE"] = np.nanmean(list(jsd_results.values())) # In case column to nan

    print("\nJensen-Shannon Divergences:")
    for col, jsd_score in jsd_results.items():
        print(col, round(jsd_score, 4))

    norm_corr_diff = syndat.metrics.normalized_correlation_difference(
        real=real_df, synthetic=synthetic_df, ignore_cat=True,
    )
    print(f"\nNormalized correlation difference: {round(norm_corr_diff, 4)}")

    if len(synthetic_df) < 2:
        print("Discriminator AUC cannot be calculated with fewer than 2 synthetic samples.")
        discrim_auc = None

    else:
        discrim_auc = syndat.metrics.discriminator_auc(real=real_df, synthetic=synthetic_df)
        print(f"\nDiscriminator AUC: {round(discrim_auc, 4)}")

    return {
        "jsd": jsd_results,
        "norm_corr_diff": norm_corr_diff,
        "discrim_auc": discrim_auc,
    }


def show_anonymeter_warning(output:str) -> None:
    """Show Anonymeter UserWarning initially and add an expandable box to allow for more information
    to be visible.

    Args:
        output (str): Function output warning message.
    """
    start = output.find("UserWarning:")

    if start != -1:
        warning_summary = output[start:]

        with st.expander(warning_summary):
            st.warning(output)


def style_status(value: str) -> str:
    """Attribute cell status marker for Jensen-Shannon Divergence (JSD) to a specific color. 

    Args:
        value (str): JSD value.

    Returns:
        str: Color specifications.
    """
    if value == "High":
        return "background-color:#fee2e2; color:#991b1b; font-weight:600;"
    if value == "Moderate":
        return "background-color:#fef3c7; color:#92400e; font-weight:600;"
    if value == "Low":
        return "background-color:#dcfce7; color:#166534; font-weight:600;"
    return ""


def calculate_metrics_ui(real_df:pd.DataFrame, synthetic_df:pd.DataFrame) -> dict[str, Any]:
    """Calculate metrics comparing the original dataframe (optionally, with missing value markers
     removed) with the synthetic dataframe (UI).

    Calculates SYNDAT metrics:
    - Jensen-Shannon Divergence
    - Normalized Correlation Difference
    - Discriminator AUC

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synthetic_df (pd.DataFrame): The synthetically generated dataframe.

    Returns:
        dict: The decision plan based on the agent's assessment.
    """
    st.subheader("Fidelity and utility metrics")
    st.write(
        "The synthetic data should resemble the original dataset. We record the following metrics "
        "from syndat to determine the dataset's fidelity:\n- Jensen-Shannon Divergence\n- "
        "Normalized correlation difference\n- Discriminator AUC"
    )
    st.write("All of these metrics output a value between 0 and 1.")

    jsd_output = StreamlitOutput(live=False)
    norm_corr_diff_output = StreamlitOutput(live=False)
    discrim_auc_output = StreamlitOutput(live=False)

    with redirect_stdout(jsd_output), redirect_stderr(jsd_output):
        jsd_results = syndat.metrics.jensen_shannon_distance(real=real_df, synthetic=synthetic_df)

    jsd_results["jsd_AVERAGE"] = np.nanmean(list(jsd_results.values())) # In case column to nan

    st.write("#### Jensen-Shannon Divergences:")
    st.write(
        "Jensen-Shannon Divergence (JSD) quantifies the statistical distributions of each variable "
        "across your real and the synthetic dataset. A value that is closer to 0 indicates that "
        "the statistical distributions identically match the real and synthetic datasets, which is "
        "favorable."
    )

    jsd_table = pd.DataFrame({
        "Column": jsd_results.keys(),
        "JSD": jsd_results.values(),
    })

    jsd_table["Status"] = jsd_table["JSD"].apply(
        lambda x: "High" if x > 0.5
        else "Low" if x <= 0.1
        else "Moderate"
    )

    styled_jsd_table = (
        jsd_table.style
        .map(style_status, subset=["Status"])
        .format({"JSD": "{:.4f}"})
    )

    df_formatter_st(styled_jsd_table)

    if jsd_output.buffer:
        with st.expander("UserWarning: Some variables may have been excluded for computation."):
            st.warning(jsd_output.buffer)

    with redirect_stdout(norm_corr_diff_output), redirect_stderr(norm_corr_diff_output):
        norm_corr_diff = syndat.metrics.normalized_correlation_difference(
            real=real_df, synthetic=synthetic_df, ignore_cat=True,
        )

    st.write("#### Normalized correlation difference:")
    st.write(
        "Normalized correlation difference calculates the correlation similarity between real and "
        "synthetic datasets and compares their correlation matrices. A value closer to 0 indicates "
        "that the synthetic dataset has greater preservation of the original correlation structure"
        "\n- Note: It is possible for the norm to exceed 1 when one dataset's correlations is "
        "positive and the other negative."
    )
    st.write(f"{round(norm_corr_diff, 4)}")

    if norm_corr_diff_output.buffer:
        with st.expander("UserWarning: Syndat normalized correlation difference output a message."):
            st.warning(norm_corr_diff_output.buffer)

    if len(synthetic_df) < 2:
        st.warning("Discriminator AUC cannot be calculated with fewer than 2 synthetic samples.")
        discrim_auc = None

        SyndatOutputs(
            jsd=jsd_output.buffer or None,
            norm_corr_diff=norm_corr_diff_output.buffer or None,
            discrim_auc=discrim_auc_output.buffer or None,
        )

    else:
        with redirect_stdout(discrim_auc_output), redirect_stderr(discrim_auc_output):
            discrim_auc = syndat.metrics.discriminator_auc(real=real_df, synthetic=synthetic_df)

        st.write("#### Discriminator AUC:")
        st.write(
            "Discriminator Area Under the Curve (AUC) is a value between 0 and 1. An AUC near 0.5 "
            "is preferable, whereas one near 1 indicates that real and synthetic datasets are easy "
            "to distinguish between."
        )
        st.write(f"{round(discrim_auc, 4)}")

        if discrim_auc_output.buffer:
            with st.expander(
                "UserWarning: A message has been output by syndat discrimination AUC tool"
            ):
                st.warning(discrim_auc_output.buffer)

        SyndatOutputs(
            jsd=jsd_output.buffer or None,
            norm_corr_diff=norm_corr_diff_output.buffer or None,
            discrim_auc=discrim_auc_output.buffer or None,
        )

    return {
        "jsd": jsd_results,
        "norm_corr_diff": norm_corr_diff,
        "discrim_auc": discrim_auc,
    }


def calculate_attack_metrics(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        synthetic_df:pd.DataFrame,
        semantic_types:dict[str, str],
        dtypes:dict[str, str],
        agent:Optional[CompiledStateGraph] | None = None,
        config:dict | None = None
    ) -> AttackMetrics:
    """Calculate Anonymeter metrics with agent assistance for column selection.
    
    Args:
        train_df (pd.DataFrame): Training portion of the real dataset.
        test_df (pd.DataFrame): Test portion of the real dataset.
        synthetic_df (pd.DataFrame): Synthetically generated dataset.
        semantic_types (dict[str, str]): Semantic types assigned to each column.
        dtypes (dict[str, str]): Python storage dtypes for each column.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to select
                        variables privacy metric calculation. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation. 
                        Defaults to None.

    Returns:
        AttackMetrics: Anonymeter privacy metric output (e.g., singling out, linkability, and 
        inference risk).
    """
    has_missing = train_df.isna().any().any() or test_df.isna().any().any()

    if has_missing: # Temporarily encode missing values as a unique marker.
        anonymeter_input = prepare_anonymeter_input(
            train_df=train_df,
            test_df=test_df,
            synthetic_df=synthetic_df,
            semantic_types=semantic_types,
            dtypes=dtypes,
        )

        train_df = anonymeter_input.train_df
        test_df = anonymeter_input.test_df
        synthetic_df = anonymeter_input.synthetic_df

    print("\nEvaluating singling out risk...")
    singling_out = SinglingOutEvaluator(
        ori=train_df,
        syn=synthetic_df,
        max_attempts=1_000,
        control=test_df,
    )
    singling_out.evaluate()
    singling_out_risk = singling_out.risk()
    print(f"\n=> Singling out risk:\n  - Value: {round(singling_out_risk.value*100, 3)}%" \
        f"\n  - Confidence interval between: {round(singling_out_risk.ci[0]*100,3)}%" \
        f" and {round(singling_out_risk.ci[1]*100,3)}%")

    prompt = attack_column_selection_message(semantic_types=semantic_types)
    print("\nAgent is selecting columns for the following privacy metrics to be assessed:" \
        "\n- Inference risk\n- Linkability risk")

    raw_response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt,}]},
        config=config,
    )

    plan = extract_output(raw_response)

    selected_cols = parse_json(plan, AttackColumns, agent, config)

    print(f"\nSelected columns:\n- Linkability:\n  - Auxiliary columns: " \
        f"{selected_cols.linkability_aux_cols}\n- Inference:\n  - Auxiliary" \
        f" columns: {selected_cols.inference_aux_cols}\n  - Secret " \
        f"column: {selected_cols.secret}\n\nRationale:")
    format_response(selected_cols.rationale)

    print("\nEvaluating inference risk...")
    inference = InferenceEvaluator(
        ori=train_df,
        syn=synthetic_df,
        aux_cols=selected_cols.inference_aux_cols,
        secret=selected_cols.secret,
        n_attacks=min(500, len(train_df), len(test_df), len(synthetic_df),),
        control=test_df,
    )

    inference.evaluate()
    inference_risk = inference.risk()
    print(f"\n=> Inference out risk:\n  - Value: {round(inference_risk.value*100, 3)}%" \
        f"\n  - Confidence interval: between {round(inference_risk.ci[0]*100,3)}%" \
        f" and {round(inference_risk.ci[1]*100,3)}%")

    print("\nEvaluating linkability risk...")
    linkability = LinkabilityEvaluator(
        ori=train_df,
        syn=synthetic_df,
        aux_cols=selected_cols.linkability_aux_cols,
        n_attacks=min(500, len(train_df), len(test_df), len(synthetic_df),),
        control=test_df,
    )

    linkability.evaluate()
    linkability_risk = linkability.risk()
    print(f"\n=> Linkability risk:\n  - Value: {round(linkability_risk.value*100, 3)}%" \
        f"\n  - Confidence interval: between {round(linkability_risk.ci[0]*100,3)}%" \
        f" and {round(linkability_risk.ci[1]*100,3)}%")

    outputs = AttackOutputs(
        singling_out=so_risk_output.buffer or None,
        inference=inference_risk_output.buffer or None,
        linkability=link_risk_output.buffer or None,
    )

    # TODO: include in report

    return AttackMetrics(
        singling_out= singling_out_risk,
        inference=inference_risk,
        linkability=linkability_risk,
        selected_cols=selected_cols,
        output_messages=outputs,
    )


def calculate_attack_metrics_ui(
    train_df:pd.DataFrame,
    test_df:pd.DataFrame,
    synthetic_df:pd.DataFrame,
    semantic_types:dict[str, str],
    dtypes:dict[str, str],
    agent:Optional[CompiledStateGraph] | None = None,
    config:dict | None = None,
) -> AttackMetrics:
    """Calculate Anonymeter metrics with agent assistance for column selection (UI).

    Args:
        train_df (pd.DataFrame): Training portion of the real dataset.
        test_df (pd.DataFrame): Test portion of the real dataset.
        synthetic_df (pd.DataFrame): Synthetically generated dataset.
        semantic_types (dict[str, str]): Semantic types assigned to each column.
        dtypes (dict[str, str]): Python storage dtypes for each column.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to select
                        variables privacy metric calculation. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation. 
                        Defaults to None.

    Returns:
        AttackMetrics: Anonymeter privacy metric output (e.g., singling out, linkability, and 
        inference risk).
    """
    st.subheader("Privacy metrics")
    st.write(
        "The synthetic dataset should also preserve privacy such that no sensitive information can "
        "be derived nor determined from the dataset. We record the following metrics from "
        "anonymeter to determine the dataset's privacy."
    )

    anonymeter_input = prepare_anonymeter_input(
        train_df=train_df,
        test_df=test_df,
        synthetic_df=synthetic_df,
        semantic_types=semantic_types,
        dtypes=dtypes,
    )

    train_df = anonymeter_input.train_df
    test_df = anonymeter_input.test_df
    synthetic_df = anonymeter_input.synthetic_df

    so_risk_output = StreamlitOutput(live=False)
    link_risk_output = StreamlitOutput(live=False)
    inference_risk_output = StreamlitOutput(live=False)

    with st.spinner("Evaluating singling out risk..."):
        with redirect_stdout(so_risk_output), redirect_stderr(so_risk_output):
            singling_out = SinglingOutEvaluator(
                ori=train_df,
                syn=synthetic_df,
                max_attempts=1_000,
                control=test_df,
            )

            singling_out.evaluate()
            singling_out_risk = singling_out.risk()

    write_attack_metric_ui(
        metric_name="Singling out",
        metric=singling_out_risk,
        description=anonymeter_metric_descriptions["singling_out"],
    )

    if so_risk_output.buffer:
        show_anonymeter_warning(so_risk_output.buffer)

    prompt = attack_column_selection_message(semantic_types=semantic_types)
    with st.spinner("Agent is selecting columns to calculate metrics..."):
        raw_response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt,}]}, config=config,
        )

        plan = extract_output(raw_response)

        selected_cols = parse_json_ui(plan, AttackColumns, agent, config,)

    info_helper_message(
        message="Agent has selected columns for inference and linkability risk assessment.",
        context="The agent was reminded of the variable names and their semantic types.",
        key="privacy_metric_column_selection",
    )

    st.write(
        "\nSelected columns:\n- Linkability:\n  - Auxiliary columns: "
        f"{selected_cols.linkability_aux_cols}\n- Inference:\n  - Auxiliary"
        f" columns: {selected_cols.inference_aux_cols}\n  - Secret "
        f"column: {selected_cols.secret}\n\nRationale:"
    )
    st.write(selected_cols.rationale)

    with st.spinner("Evaluating inference risk..."):
        with redirect_stdout(inference_risk_output), redirect_stderr(inference_risk_output):
            inference = InferenceEvaluator(
                ori=train_df,
                syn=synthetic_df,
                aux_cols=selected_cols.inference_aux_cols,
                secret=selected_cols.secret,
                n_attacks=min(500, len(train_df), len(test_df), len(synthetic_df),),
                control=test_df,
            )

            inference.evaluate()
            inference_risk = inference.risk()

    write_attack_metric_ui(
        metric_name="Inference",
        metric=inference_risk,
        description=anonymeter_metric_descriptions["inference"]
    )

    if inference_risk_output.buffer:
        show_anonymeter_warning(inference_risk_output.buffer)

    with st.spinner("Evaluating linkability risk..."):
        with redirect_stdout(link_risk_output), redirect_stderr(link_risk_output):
            linkability = LinkabilityEvaluator(
                ori=train_df,
                syn=synthetic_df,
                aux_cols=selected_cols.linkability_aux_cols,
                n_attacks=min(500, len(train_df), len(test_df), len(synthetic_df),),
                control=test_df,
            )

            linkability.evaluate()
            linkability_risk = linkability.risk()

    write_attack_metric_ui(
        metric_name="Linkability",
        metric=linkability_risk,
        description=anonymeter_metric_descriptions["linkability"],
    )

    if link_risk_output.buffer:
        show_anonymeter_warning(link_risk_output.buffer)

    outputs = AttackOutputs(
        singling_out=so_risk_output.buffer or None,
        inference=inference_risk_output.buffer or None,
        linkability=link_risk_output.buffer or None,
    )
    # TODO: Tie outputs in with the report

    return AttackMetrics(
        singling_out= singling_out_risk,
        inference=inference_risk,
        linkability=linkability_risk,
        selected_cols=selected_cols,
        output_messages=outputs,
    )


def display_attack_metrics_ui(attack_metrics: AttackMetrics, description:dict[str, str]) -> None:
    """Display what was calculated from Anonymeter's functions and the variables that were used
    to calculate the metrics. This is done because, unlike syndat, these functions take longer to 
    run.

    Args:
        attack_metrics (AttackMetrics): Stores information regarding privacy metrics to be 
                                        displayed.
        description (dict[str, str]): Quick overview of attack metrics.
    """
    st.subheader("Privacy metrics")
    st.write(
        "The synthetic dataset should also preserve privacy such that no sensitive information can "
        "be derived nor determined from the dataset. We record the following metrics from "
        "anonymeter to determine the dataset's privacy."
    )

    singling_out_risk = attack_metrics.singling_out
    inference_risk = attack_metrics.inference
    linkability_risk = attack_metrics.linkability
    selected_cols = attack_metrics.selected_cols

    outputs = attack_metrics.output_messages

    write_attack_metric_ui(
        metric_name="Singling out",
        metric=singling_out_risk,
        description=description["singling_out"],
    )

    if outputs.singling_out:
        show_anonymeter_warning(outputs.singling_out)

    info_helper_message(
        message="Agent has selected columns for inference and linkability risk assessment.",
        context="The agent was reminded of the variable names and their semantic types.",
        key="privacy_metric_column_selection_displayed",
    )

    st.write(
        "\nSelected columns:\n- Linkability:\n  - Auxiliary columns: "
        f"{selected_cols.linkability_aux_cols}\n- Inference:\n  - Auxiliary"
        f" columns: {selected_cols.inference_aux_cols}\n  - Secret "
        f"column: {selected_cols.secret}\n\nRationale:"
    )

    st.write(selected_cols.rationale)

    write_attack_metric_ui(
        metric_name="Inference",
        metric=inference_risk,
        description=description["inference"],
    )

    if outputs.inference:
        show_anonymeter_warning(outputs.inference)

    write_attack_metric_ui(
        metric_name="Linkability",
        metric=linkability_risk,
        description=description["linkability"],
    )

    if outputs.linkability:
        show_anonymeter_warning(outputs.linkability)

def write_attack_metric_ui(metric_name:str, metric:"PrivacyRisk", description:str) -> None:
    """Display privacy metric information.

    Args:
        metric_name (str): Name of the metric (for display).
        metric (PrivacyRisk): Details (e.g., value and confidence interval) for each metric.
        description (str): Basic explanation of the metric for a quick overview.
    """
    st.write(
        f"#### {metric_name} risk:\n{description}\n - Value: {round(metric.value*100, 3)}%\n  - "
        f"Confidence interval: between {round(metric.ci[0]*100,3)}% and {round(metric.ci[1]*100,3)}"
        "%"
    )


def value_exists(candidate: object, *dfs: pd.DataFrame) -> bool:
    """Return whether a value occurs anywhere in any supplied dataframe."""
    return any(
        df.eq(candidate).any().any()
        for df in dfs
    )


def find_unique_string(*dfs: pd.DataFrame) -> str:
    """" Find a unique string that can be used a fill in for missing values
    such that anonymeter may be applied.

    Returns:
        str: Missing value marker applied.
    """
    base = "__missing__"
    candidate = base
    counter = 1

    while value_exists(candidate, *dfs):
        candidate = f"{base}_{counter}"
        counter += 1

    return candidate


def find_unique_integer(*dfs: pd.DataFrame) -> int:
    """" Find a unique integer that can be used a fill in for missing values
    such that anonymeter may be applied.
    
    Returns:
        str: Missing value marker applied.
    """
    candidate = -999

    while value_exists(candidate, *dfs):
        candidate -= 1

    return candidate


def find_unique_float(*dfs: pd.DataFrame) -> float:
    """" Find a unique float that can be used a fill in for missing values
    such that anonymeter may be applied.
    
    Returns:
        str: Missing value marker applied.
    """
    candidate = -999.5

    while value_exists(candidate, *dfs):
        candidate -= 1

    return candidate


def select_missing_marker(
    semantic_type:str,
    dtype:str,
    *dfs:pd.DataFrame,
) -> str | int | float:
    """Select a unique missing value marker that is compatible with
    the column's semantic type and Python storage dtype.

    Args:
        semantic_type (str): Integer, float, categorical, etc.
        dtype (str): Python storage dtype (default is majority).

    Returns:
        str | int | float: A unique value that is not present in the
        dataframes provided.
    """
    if semantic_type == "Integer":
        return find_unique_integer(*dfs)

    if semantic_type == "Float":
        return find_unique_float(*dfs)

    if semantic_type in {"Categorical", "Ordinal"}:
        if dtype == "Int64":
            return find_unique_integer(*dfs)
        if dtype == "string":
            return find_unique_string(*dfs)

    if semantic_type == "Boolean":
        return find_unique_string(*dfs)


def prepare_anonymeter_input(
    train_df:pd.DataFrame,
    test_df:pd.DataFrame,
    synthetic_df:pd.DataFrame,
    semantic_types:dict[str, str],
    dtypes:dict[str, str],
) -> AnonymeterInput:
    """Prepare copies of train, test, and synthetic dataframes for Anonymeter evaluation
    in the event that there are missing values. Missing values are temporarily replaced
    with unique markers that are compatible with each column's data type.

    Args:
        train_df (pd.DataFrame): Training portion of the real dataset.
        test_df (pd.DataFrame): Test portion of the real dataset.
        synthetic_df (pd.DataFrame): Synthetically generated dataset.
        semantic_types (dict[str, str]): Semantic types assigned to each column.
        dtypes (dict[str, str]): Python storage dtypes for each column.

    Returns:
        AnonymeterInput: Prepared dataframe copies and the missing value markers applied to
        each column.
    """
    train = train_df.copy()
    test = test_df.copy()
    synthetic = synthetic_df.copy()

    frames = (train, test, synthetic)
    markers:dict[str, str | int | float] = {}

    for col, semantic_type in semantic_types.items():
        if any(df[col].isna().any() for df in frames):
            marker = select_missing_marker(semantic_type, dtypes[col], *frames,)
            markers[col] = marker

            for df in frames:
                df[col] = df[col].fillna(marker)

    for col, dtype in dtypes.items():
        for df in frames:
            if semantic_types[col] == "Boolean":
                df[col] = df[col].astype(str)

            elif dtype in {"Int64", "int64"}:
                df[col] = pd.to_numeric(df[col]).astype("int64")

            elif dtype == "float64":
                df[col] = df[col].astype("float64")

            elif dtype in {"string", "object"}:
                df[col] = df[col].astype(str)

    return AnonymeterInput(
        train_df=train,
        test_df=test,
        synthetic_df=synthetic,
        missing_markers=markers,
    )


def generate_plots(real_df:pd.DataFrame, synth_df:pd.DataFrame, version_id:int) -> list[Path]:
    """Plot feature distributions and store files.

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                        missing value markers replaced with nan (norm_na_df).
        synth_df (pd.DataFrame): The synthetically generated dataframe.
        version_id (int): Denotes the version of the workflow attempt.

    Returns:
        list[Path]: JSD plot images.
    """
    # TODO: verify whether path needs to be fixed here too
    output_dir = Path("../output") / f"version_{version_id}" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    syndat.visualization.plot_distributions(real_df, synth_df, store_destination=str(output_dir))

    png_files = sorted(output_dir.glob("*.png"))

    print(f"\nDistribution plot for: {png_files[0].name}")
    display(Image(filename=str(png_files[0])))

    print(f"\n=> Distribution plots for all features saved to: {output_dir}")

    return png_files


def encode_for_correlation(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, max_categories: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categorical features for correlation analysis while limiting feature expansion.

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synth_df (pd.DataFrame): The synthetically generated dataframe.
        max_categories (int, optional): _description_. Defaults to 10.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    n_real = len(real_df)
    combined = pd.concat([real_df, synth_df], ignore_index=True,)
    combined = combined.loc[:, ~combined.columns.str.startswith("Unnamed:")]

    categorical_cols = combined.select_dtypes(
        include=["object", "str", "category"]
    ).columns

    low_cardinality = [
        col for col in categorical_cols if combined[col].nunique(dropna=True) <= max_categories
    ]
    high_cardinality = [col for col in categorical_cols if col not in low_cardinality]

    # One-hot only manageable categorical variables
    encoded = pd.get_dummies(combined, columns=low_cardinality, dtype=int,)

    # Exclude high-cardinality categoricals from Pearson correlation
    encoded = encoded.drop(columns=high_cardinality)

    # TODO: add a limit to the encoding (perhaps preferring first to encode smaller card. cols first)

    return (encoded.iloc[:n_real], encoded.iloc[n_real:],)


def encode_for_shap(
    real_df: pd.DataFrame, synth_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categorical features numerically for SHAP discrimination analysis.

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synth_df (pd.DataFrame): The synthetically generated dataframe.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    n_real = len(real_df)
    combined = pd.concat([real_df, synth_df], ignore_index=True,)
    combined = combined.loc[:, ~combined.columns.str.startswith("Unnamed:")]

    categorical_cols = combined.select_dtypes(include=["object", "str", "category"]).columns

    for col in categorical_cols:
        combined[col], _ = pd.factorize(combined[col])

    return (combined.iloc[:n_real], combined.iloc[n_real:],)



@st.fragment
def plot_viewer(images, plot_type:str, version_id:int, show_filename=False,) -> None:
    """Display plot images with previous and next navigation controls.

    Args:
        images (_type_): Cached plot images to display.
        plot_type (str): Plot type used to distinguish session-state and widget keys.
        version_id (int): Denotes the version of the workflow attempt.
        show_filename (bool): Whether filename should be displayed. Defaults to False.
    """
    index_key = f"{plot_type}_figure_idx_{version_id}"

    if index_key not in st.session_state:
        st.session_state[index_key] = 0

    st.session_state[index_key] = min(
        st.session_state[index_key],
        len(images) - 1,
    )

    idx = st.session_state[index_key]

    col_prev, col_count, col_next = st.columns([1, 8, 1])

    with col_prev:
        if st.button(
            "←",
            disabled=idx == 0,
            key=f"previous_{plot_type}_{version_id}",
        ):
            st.session_state[index_key] -= 1
            st.rerun(scope="fragment")

    with col_count:
        st.markdown(
            f"<div style='text-align:center'>{idx + 1} / {len(images)}</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        if st.button(
            "→",
            disabled=idx == len(images) - 1,
            key=f"next_{plot_type}_{version_id}",
        ):
            st.session_state[index_key] += 1
            st.rerun(scope="fragment")

    idx = st.session_state[index_key]
    if show_filename:
        st.caption(Path(images[idx]["name"]).name)
    st.image(images[idx]["data"], width="stretch")


def _cache_pngs(directory: Path) -> list[dict]:
    """Load PNG files from a directory into memory

    Args:
        directory (Path): Directory containing PNG files.

    Returns:
        list[dict]: Cached PNG data and filenames.
    """
    return [
        {"name": path.name, "data": path.read_bytes()} for path in sorted(directory.glob("*.png"))
    ]


def _generate_plots(real_df:pd.DataFrame, synth_df:pd.DataFrame, output_dir:Path) -> dict:
    """Generate and cache distribution, correlation, and SHAP plots.

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synth_df (pd.DataFrame): The synthetically generated dataframe.
        output_dir (Path): Directory for storing generated plots.

    Returns:
        dict: Cached plot images grouped by plot type.
    """
    distribution_dir = output_dir / "distributions"
    correlation_dir = output_dir / "correlations"
    shap_dir = output_dir / "shap"

    for directory in [distribution_dir, correlation_dir, shap_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Distribution plots
    syndat.visualization.plot_distributions(
        real=real_df,
        synthetic=synth_df,
        store_destination=str(distribution_dir),
    )

    # Correlation plots
    real_corr, syn_corr = encode_for_correlation(real_df, synth_df)
    syndat.visualization.plot_correlations(
        real=real_corr,
        synthetic=syn_corr,
        store_destination=str(correlation_dir),
    )

    # SHAP discrimination plot
    real_shap, syn_shap = encode_for_shap(real_df, synth_df)
    shap_path = shap_dir / "shap_discrimination.png"

    syndat.visualization.plot_shap_discrimination(
        real=real_shap,
        synthetic=syn_shap,
        save_path=str(shap_path),
    )

    return {
        "distributions": _cache_pngs(distribution_dir),
        "correlations": _cache_pngs(correlation_dir),
        "shap": _cache_pngs(shap_dir),
    }


def generate_plots_ui(real_df:pd.DataFrame, synth_df:pd.DataFrame, version_id:int) -> dict:
    """Generate, cache, store, and display distribution, correlation, and SHAP plots (UI).

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                        missing value markers replaced with nan (norm_na_df).
        synth_df (pd.DataFrame): The synthetically generated dataframe.
        version_id (int): Denotes the version of the workflow attempt.

    Returns:
        dict: Cached plot images grouped by plot type.
    """
    output_dir = Path("output") / f"version_{version_id}" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_key = f"plots_{version_id}"

    if cache_key not in st.session_state:
        st.session_state[cache_key] = _generate_plots(real_df, synth_df, output_dir)

    plots = st.session_state[cache_key]

    tab_dist, tab_corr, tab_shap = st.tabs([
        "Distribution plots",
        "Correlation plots",
        "SHAP discrimination plot",
    ])

    with tab_dist:
        plot_viewer(
            images=plots["distributions"],
            plot_type="distribution",
            version_id=version_id,
        )

    with tab_corr:
        st.write("Correlation plots")
        plot_viewer(
            images=plots["correlations"],
            plot_type="correlation",
            version_id=version_id,
            show_filename=True,
        )

    with tab_shap:
        st.write("SHAP plot")
        plot_viewer(
            images=plots["shap"],
            plot_type="shap",
            version_id=version_id,
        )

    return plots


def evaluation(
        synthesis_request:SynthesisRequest,
        metrics:dict[str, Any],
        attack_metrics:dict[str, Any],
        agent:Optional[CompiledStateGraph] | None = None, 
        config:dict | None = None
    ) -> AgentEval:
    """The agent evaluates the synthetic dataset using calculated metrics.

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synthetic_df (pd.DataFrame): The synthetically generated dataframe.
        synthesis_request (SynthesisRequest): Dictionary of the user's preferences. This is used as
                                a record of the processing done to generate the synthetic data.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to evaluate the
                                synthetic data based on SYNDAT metrics. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation. 
                                Defaults to None.

    Returns:
        AgentEval: The decision plan based on the agent's assessment.
    """
    request_dict = synthesis_request.model_dump()
    prompt = evaluation_prompt(
        synthesis_request=request_dict, metrics=metrics, attack_metrics=attack_metrics
    )

    print("\nAgent is evaluating the results...")

    raw_response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt,}]},
        config=config,
    )

    plan = extract_output(raw_response)

    eval_plan = parse_json(plan, AgentEval, agent, config)

    print("\nEvaluation:")
    print(f"- Status: {eval_plan.status}")
    if len(eval_plan.problem_columns) > 0:
        print(f"- Problem columns: {eval_plan.problem_columns}")
    print("- Recommendation:")
    format_response(eval_plan.recommendation)
    print("- Rationale:")
    format_response(eval_plan.rationale)
    return eval_plan


def evaluation_ui(
    synthesis_request:SynthesisRequest,
    metrics:dict[str, Any],
    attack_metrics:dict[str, Any],
    agent:Optional[CompiledStateGraph] | None = None,
    config:dict | None = None,
) -> AgentEval:
    """The agent evaluates the synthetic dataset using calculated metrics (UI).

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synthetic_df (pd.DataFrame): The synthetically generated dataframe.
        synthesis_request (SynthesisRequest): Dictionary of the user's preferences. This is used as
                                a record of the processing done to generate the synthetic data.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to evaluate the
                                synthetic data based on SYNDAT metrics. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation. 
                                Defaults to None.

    Returns:
        AgentEval: The decision plan based on the agent's assessment.
    """
    request_dict = synthesis_request.model_dump()
    prompt = evaluation_prompt(
        synthesis_request=request_dict,
        metrics=metrics,
        attack_metrics=attack_metrics,
    )

    if "agent_evaluation" not in st.session_state:
        with st.spinner("Agent is evaluating the results..."):
            raw_response = agent.invoke(
                {"messages": [{"role": "user", "content": prompt,}]},
                config=config,
            )

            plan = extract_output(raw_response)

            eval_plan = parse_json_ui(plan, AgentEval, agent, config,)

        st.session_state.agent_evaluation = eval_plan

    else:
        eval_plan = st.session_state.agent_evaluation

    info_helper_message(
        message="The agent has finished its evaluation of the synthetic dataset.",
        context=(
            "The agent evaluated the synthetic dataset based on:\n- Your preferences.\n- "
            "Calculated syndat metrics.\n- Calculated Anonymeter metrics."
        ),
        key="agent_eval_synth_data_info",
    )

    st.subheader("Evaluation:")
    st.write(f"- Status: {eval_plan.status}")
    if len(eval_plan.problem_columns) > 0:
        st.write(f"- Problem columns: {eval_plan.problem_columns}")
    st.write("#### - Recommendation:")
    st.write(eval_plan.recommendation)
    st.write("#### - Rationale:")
    st.write(eval_plan.rationale)
    return eval_plan


def evaluation_review(
        agent_decision:AgentEval, 
        agent:Optional[CompiledStateGraph] | None = None, 
        config:dict | None = None
        ) -> str:
    """User asks questions about agent's assessment/discusses details to focus on.

    Args:
        agent_decision (AgentEval): Agent's evaluation decision.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to review the 
                                                               evaluation. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation.
                                        Defaults to None.

    Returns:
        str: Whether the workflow should finish or be revised.
    """
    confirm = pyip.inputYesNo("\nDo you have questions about the evaluation results?\n")

    while confirm == "yes":
        response = pyip.inputStr("\nEnter your question/concern:\n")
        print("\nAgent is preparing its response...")
        prompt = eval_review_message(agent_decision, response)
        raw_response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt,}]},
            config=config,
        )

        format_response(extract_output(raw_response))

        confirm = pyip.inputYesNo("\nDo you have another question?\n")

    print(f"\nWorkflow result status: {agent_decision.status}")
    choice = pyip.inputMenu(
        ["finish", "revise"],
        "\nWould you like to finish or revise the workflow?\n", 
        numbered=True
    )

    return choice


def evaluation_review_ui(
    agent_decision:AgentEval,
    agent:Optional[CompiledStateGraph] | None = None,
    config:dict | None = None,
    version_id:int | None = None,
) -> str:
    """User asks questions about agent's assessment/discusses details to focus on (UI).

    Args:
        agent_decision (AgentEval): Agent's evaluation decision.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to review the 
                                                               evaluation. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation.
                                        Defaults to None.

    Returns:
        str: Whether the workflow should finish or be revised.
    """
    if "evaluation_review_history" not in st.session_state:
        st.session_state.evaluation_review_history = []

    confirm = st.selectbox(
        "Do you have questions about the evaluation results?",
        ["yes", "no"],
        index=None,
        key=f"evaluation_review_confirm_{version_id}",
    )

    if confirm is None:
        st.stop()

    if confirm == "yes":
        for item in st.session_state.evaluation_review_history:
            with st.chat_message("user"):
                st.write(item["question"])

            with st.chat_message("assistant"):
                st.write(item["answer"])

        new_question = st.chat_input(
            "Enter your question/concern.",
            key=f"evaluation_review_input_{version_id}",
        )

        answer = ""

        if new_question:
            with st.spinner("\nAgent is preparing its response..."):
                prompt = eval_review_message(agent_decision, new_question)
                raw_response = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt,}]},
                    config=config,
                )

            answer = st.write(extract_output(raw_response))

        st.session_state.evaluation_review_history.append({
            "question": new_question,
            "answer": answer,
        })

        st.rerun()

        finished = st.button(
            "Done asking questions",
            key=f"evaluation_review_done_{version_id}",
        )

        if not finished:
            st.stop()

    # If no is selected:
    st.write(f"Workflow result status: {agent_decision.status}")

    choice = st.selectbox(
        "Would you like to finish or revise the workflow?",
        ["finish", "revise"],
        index=None,
        key=f"evaluation_workflow_choice_{version_id}",
    )

    if choice is None:
        st.stop()

    return choice


def write_report(report: Report, synthesis_request:SynthesisRequest, data_path: str, version_id:int) -> None:
    """Write a workflow report based on decisions made during the synthetic data generation process.

    Args:
        report (Report): Dictionary containing information gathered throughout the workflow steps.
        synthesis_request (SynthesisRequest): Dictionary with the user's preferences.
        data_path (str): Initial path of the data (used for naming the report).
        version_id (int): Denotes the version of the workflow attempt.
    """
    request_dict = synthesis_request.model_dump()

    data_name = Path(data_path).stem
    output_filename = f"report_{data_name}_v{version_id}.md"
    output_path = Path("output") / f"version_{version_id}" / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# SYNTHETIC DATA GENERATION REPORT\n")

        f.write("## Input Dataset:\n")
        f.write(f"- Path: {report.input.path}\n")
        f.write(f"- Original shape: {report.input.original_shape}\n")
        f.write(f"- Prepared shape: {report.input.prepared_shape}\n\n")

        f.write("## Preprocessing:\n")
        f.write("Missing Value Detection\n")
        f.write("Detected markers:\n")
        f.write("```python\n")
        pprint.pprint(report.prep.missing_markers_detected, stream=f, sort_dicts=False,)

        f.write("\n Replaced markers:\n")
        f.write("```python\n")
        pprint.pprint(report.prep.missing_markers_replaced, stream=f, sort_dicts=False,)
        f.write("```\n")

        f.write("\n\n### Python data types\n")
        f.write("```python\n")
        pprint.pprint(report.prep.dtypes, stream=f, sort_dicts=False)
        f.write("```\n")

        f.write("\n\n### Semantic data types\n")
        f.write("```python\n")
        pprint.pprint(report.prep.stypes, stream=f, sort_dicts=False)
        f.write("```\n")

        f.write("\n\n### Missing Value Handling\n")
        f.write("```python\n")
        pprint.pprint(report.prep.missing_before, stream=f, sort_dicts=False,)
        pprint.pprint(report.prep.missing_after, stream=f, sort_dicts=False,)
        f.write("```\n")

        f.write("\n\n### Generation:\n")
        f.write(f"Chosen generator: {request_dict['generator']}\n")
        f.write(f"Prepared dataset: {report.gen.prep_path}\n")
        f.write(f"Synthetic dataset: {report.gen.synth_path}\n")
        f.write(f"Synthetic shape: {report.gen.synth_shape}\n")

        f.write("\n\n### Evaluation\n")
        f.write("```python\n")
        pprint.pprint(report.review.jsd, stream=f, sort_dicts=False)
        f.write("```\n")

        f.write("\nConclusion\n")
        f.write("```python\n")
        pprint.pprint(report.review.recommendation, stream=f, sort_dicts=False,)
        f.write("```\n")

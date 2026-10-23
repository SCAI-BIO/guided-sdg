from contextlib import redirect_stderr, redirect_stdout
from typing import Optional

from anonymeter.evaluators import InferenceEvaluator, LinkabilityEvaluator, SinglingOutEvaluator
from langgraph.graph.state import CompiledStateGraph
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from agentic_synth_data.schemas.attack_model import AnonymeterInput, AttackMetrics, AttackColumns, \
    AttackOutputs
from agentic_synth_data.schemas.streamlit_helpers import CapturedOutput, StreamlitOutput
from agentic_synth_data.utils import SemanticTypes, format_response, extract_output, \
    info_helper_message, parse_json, parse_json_ui
from agentic_synth_data.prompts.evaluation_prompt import attack_column_selection_message


# ============================================================================
# Anonymeter metric description
# ============================================================================

ANONYMETER_METRIC_DESCRIPTION = {
        "singling_out": (
            "Measures whether an attacker can uniquely identify an individual "
            "record in the original dataset using information from the synthetic data."
        ),
        "linkability": (
            "Measures whether an attacker can link information about the same "
            "individual across different sets of attributes using the synthetic data."
        ),
        "inference": (
            "Measures whether an attacker can predict sensitive or unknown "
            "attribute values in the original dataset using known attributes and the synthetic "
            "data."
        ),
    }


# ============================================================================
# Anonymeter metric calculation
# ============================================================================

def _calculate_singling_out(
    train: pd.DataFrame, test: pd.DataFrame, synth: pd.DataFrame, ui: bool = False,
) -> tuple["PrivacyRisk", str | None]:
    """Calculate singling out risk and capture output messages.

    Returns:
        tuple[PrivacyRisk, str | None]: Calculated privacy risk and any captured output messages.
    """
    output = StreamlitOutput(live=False) if ui else CapturedOutput()

    with redirect_stdout(output), redirect_stderr(output):
        evaluator = SinglingOutEvaluator(
            ori=train, syn=synth, max_attempts=1_000, control=test,
        )

        evaluator.evaluate()
        risk = evaluator.risk()

    return risk, output.buffer or None


def _calculate_inference(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synth: pd.DataFrame,
    cols: AttackColumns,
    ui: bool = False,
) -> tuple["PrivacyRisk", str | None]:
    """Calculate inference risk and capture output messages.

    Returns:
        tuple[PrivacyRisk, str | None]: Calculated privacy risk and any captured output messages.
    """
    output = StreamlitOutput(live=False) if ui else CapturedOutput()

    with redirect_stdout(output), redirect_stderr(output):
        evaluator = InferenceEvaluator(
            ori=train,
            syn=synth,
            aux_cols=cols.inference_aux_cols,
            secret=cols.secret,
            n_attacks=min(500, len(train), len(test), len(synth),),
            control=test,
        )

        evaluator.evaluate()
        risk = evaluator.risk()

    return risk, output.buffer or None


def _calculate_linkability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synth: pd.DataFrame,
    cols: AttackColumns,
    ui: bool = False,
) -> tuple["PrivacyRisk", str | None]:
    """Calculate linkability risk and capture output messages.

    Returns:
        tuple[PrivacyRisk, str | None]: Calculated privacy risk and any captured output messages.
    """
    output = StreamlitOutput(live=False) if ui else CapturedOutput()

    with redirect_stdout(output), redirect_stderr(output):
        evaluator = LinkabilityEvaluator(
            ori=train,
            syn=synth,
            aux_cols=cols.linkability_aux_cols,
            n_attacks=min(500, len(train), len(test), len(synth),),
            control=test,
        )

        evaluator.evaluate()
        risk = evaluator.risk()

    return risk, output.buffer or None


# ============================================================================
# Anonymeter metric display
# ============================================================================

def calculate_attack_metrics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synth: pd.DataFrame,
    stypes: SemanticTypes,
    dtypes: dict[str, str],
    agent: Optional[CompiledStateGraph] | None = None,
    config: dict | None = None,
) -> AttackMetrics:
    """Calculate Anonymeter metrics with agent assistance for column selection.
    
    Args:
        train (pd.DataFrame): Training portion of the real dataset.
        test (pd.DataFrame): Test portion of the real dataset.
        synth (pd.DataFrame): Synthetically generated dataset.
        stypes (SemanticTypes): Semantic types assigned to each column.
        dtypes (dict[str, str]): Python storage dtypes for each column.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to select
            variables privacy metric calculation. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation. 
            Defaults to None.

    Returns:
        AttackMetrics: Anonymeter privacy metric output (e.g., singling out, linkability, and 
            inference risk).
    """
    has_missing = train.isna().any().any() or test.isna().any().any()

    if has_missing: # Temporarily encode missing values as a unique marker.
        anonymeter_input = prepare_anonymeter_input(
            train=train,
            test=test,
            synth=synth,
            stypes=stypes,
            dtypes=dtypes,
        )

        train = anonymeter_input.train
        test = anonymeter_input.test
        synth = anonymeter_input.synth

    # Calculate singling out risk:
    print("\nEvaluating singling out risk...")

    so_risk, so_risk_output = _calculate_singling_out(train, test, synth,)

    write_attack_metric(metric_name="Singling out", metric=so_risk,)

    # Select columns for inference and linkability risk:
    prompt = attack_column_selection_message(stypes=stypes)
    print("\nAgent is selecting columns for the following privacy metrics to be assessed:" \
        "\n- Inference risk\n- Linkability risk")

    raw_response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt,}]},
        config=config,
    )

    plan = extract_output(raw_response)

    cols = parse_json(plan, AttackColumns, agent, config)

    print(f"\nSelected columns:\n- Linkability:\n  - Auxiliary columns: " \
        f"{cols.linkability_aux_cols}\n- Inference:\n  - Auxiliary" \
        f" columns: {cols.inference_aux_cols}\n  - Secret " \
        f"column: {cols.secret}\n\nRationale:")
    format_response(cols.rationale)

    # Calculate inference risk:
    print("\nEvaluating inference risk...")

    inference_risk, inference_risk_output = _calculate_inference(train, test, synth, cols,)

    write_attack_metric(metric_name="Inference", metric=inference_risk,)

    # Calculate linkability risk:
    print("\nEvaluating linkability risk...")

    linkability_risk, link_risk_output = _calculate_inference(train, test, synth, cols,)

    write_attack_metric(metric_name="Linkability", metric=linkability_risk,)

    outputs = AttackOutputs(
        singling_out=so_risk_output or None,
        inference=inference_risk_output or None,
        linkability=link_risk_output or None,
    )

    return AttackMetrics(
        singling_out= so_risk,
        inference=inference_risk,
        linkability=linkability_risk,
        cols=cols,
        output_messages=outputs,
    )


def calculate_attack_metrics_ui(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synth: pd.DataFrame,
    stypes: SemanticTypes,
    dtypes: dict[str, str],
    agent: Optional[CompiledStateGraph] | None = None,
    config: dict | None = None,
) -> AttackMetrics:
    """Calculate Anonymeter metrics with agent assistance for column selection (UI).

    Args:
        train (pd.DataFrame): Training portion of the real dataset.
        test (pd.DataFrame): Test portion of the real dataset.
        synth (pd.DataFrame): Synthetically generated dataset.
        stypes (SemanticTypes): Semantic types assigned to each column.
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
        train=train,
        test=test,
        synth=synth,
        stypes=stypes,
        dtypes=dtypes,
    )

    train = anonymeter_input.train
    test = anonymeter_input.test
    synth = anonymeter_input.synth

    # Calculate singling out risk:
    with st.spinner("Evaluating singling out risk..."):
        so_risk, so_risk_output = _calculate_singling_out(train, test, synth, True,)

    write_attack_metric_ui(
        metric_name="Singling out",
        metric=so_risk,
        description=ANONYMETER_METRIC_DESCRIPTION["singling_out"],
    )

    if so_risk_output:
        show_anonymeter_warning_ui(so_risk_output)

    # Select columns for inference and linkability risk:
    prompt = attack_column_selection_message(stypes=stypes)
    with st.spinner("Agent is selecting columns to calculate metrics..."):
        raw_response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt,}]},
            config=config,
        )

        plan = extract_output(raw_response)

        cols = parse_json_ui(plan, AttackColumns, agent, config,)

    info_helper_message(
        message="Agent has selected columns for inference and linkability risk assessment.",
        context="The agent was reminded of the variable names and their semantic types.",
        key="privacy_metric_column_selection",
    )

    st.write(
        "\nSelected columns:\n"
        "- Linkability:\n"
        f"  - Auxiliary columns: {cols.linkability_aux_cols}\n"
        "- Inference:\n"
        f"  - Auxiliary columns: {cols.inference_aux_cols}\n"
        f"  - Secret column: {cols.secret}\n\n"
        "Rationale:"
    )
    st.write(cols.rationale)

    # Calculate inference risk:
    with st.spinner("Evaluating inference risk..."):
        inference_risk, inference_risk_output = _calculate_inference(train, test, synth, cols, True)

    write_attack_metric_ui(
        metric_name="Inference",
        metric=inference_risk,
        description=ANONYMETER_METRIC_DESCRIPTION["inference"]
    )

    if inference_risk_output:
        show_anonymeter_warning_ui(inference_risk_output)

    # Calculate linkability risk:
    with st.spinner("Evaluating linkability risk..."):
        linkability_risk, link_risk_output = _calculate_linkability(train, test, synth, cols, True)

    write_attack_metric_ui(
        metric_name="Linkability",
        metric=linkability_risk,
        description=ANONYMETER_METRIC_DESCRIPTION["linkability"],
    )

    if link_risk_output:
        show_anonymeter_warning_ui(link_risk_output)

    outputs = AttackOutputs(
        singling_out=so_risk_output or None,
        inference=inference_risk_output or None,
        linkability=link_risk_output or None,
    )

    return AttackMetrics(
        singling_out= so_risk,
        inference=inference_risk,
        linkability=linkability_risk,
        cols=cols,
        output_messages=outputs,
    )


def display_attack_metrics_ui(attack_metrics: AttackMetrics, description: dict[str, str]) -> None:
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
    cols = attack_metrics.cols

    outputs = attack_metrics.output_messages

    write_attack_metric_ui(
        metric_name="Singling out",
        metric=singling_out_risk,
        description=description["singling_out"],
    )

    if outputs.singling_out:
        show_anonymeter_warning_ui(outputs.singling_out)

    info_helper_message(
        message="Agent has selected columns for inference and linkability risk assessment.",
        context="The agent was reminded of the variable names and their semantic types.",
        key="privacy_metric_column_selection_displayed",
    )

    st.write(
        "\nSelected columns:\n"
        "- Linkability:\n"
        f"  - Auxiliary columns: {cols.linkability_aux_cols}\n"
        "- Inference:\n"
        f"  - Auxiliary columns: {cols.inference_aux_cols}\n"
        f"  - Secret column: {cols.secret}\n\n"
        "Rationale:"
    )

    st.write(cols.rationale)

    write_attack_metric_ui(
        metric_name="Inference", metric=inference_risk, description=description["inference"],
    )

    if outputs.inference:
        show_anonymeter_warning_ui(outputs.inference)

    write_attack_metric_ui(
        metric_name="Linkability", metric=linkability_risk, description=description["linkability"],
    )

    if outputs.linkability:
        show_anonymeter_warning_ui(outputs.linkability)


# ============================================================================
# UI and CLI helpers
# ============================================================================

def write_attack_metric(metric_name: str, metric: "PrivacyRisk",) -> None:
    """Display privacy metric information.

    Args:
        metric_name (str): Name of the metric (for display).
        metric (PrivacyRisk): Details (e.g., value and confidence interval) for each metric.
    """
    print(
        f"\n=> {metric_name} risk:\n  "
        f"- Value: {round(metric.value*100, 3)}%\n  "
        f"- Confidence interval: between {round(metric.ci[0]*100,3)}% and "
        f"{round(metric.ci[1]*100,3)}%"
    )


def show_anonymeter_warning_ui(output: str) -> None:
    """Show Anonymeter UserWarning initially and add an expandable box to allow for more information
    to be visible (UI).

    Args:
        output (str): Function output warning message.
    """
    start = output.find("UserWarning:")

    if start != -1:
        warning_summary = output[start:]

        with st.expander(warning_summary):
            st.warning(output)


def display_privacy_risk_ui(label: str, value: float, ci_lower: float, ci_upper: float,) -> None:
    """Display a privacy risk estimate and its confidence interval (UI).

    Args:
        label (str): Type of privacy risk metric.
        value (float): Resulting value.
        ci_lower (float): Lower confidence interval.
        ci_upper (float): Upper confidence interval.
    """
    st.metric(label=label, value=f"{value:.2%}",) # TODO: Can add change between versions

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[value * 100],
            y=[0],
            mode="markers",
            marker={"size": 12},
            error_x={
                "type": "data",
                "symmetric": False,
                "visible": True,
                "array": [(ci_upper - value) * 100],
                "arrayminus": [(value - ci_lower) * 100],
                "thickness": 3,
                "width": 8,
            },
            hovertemplate="Risk: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        height=160,
        margin={"l": 10, "r": 20, "t": 10, "b":35,},
        showlegend=False,
        xaxis={"title": "Privacy risk (%)", "range": [0, 100]},
        yaxis={"visible": False, "range":[-1, 1],},
    )

    st.plotly_chart(fig, width="stretch", key=f"privacy_risk_{label}",)

    st.caption(f"95% confidence interval: {ci_lower:.1%} – {ci_upper:.1%}")


def write_attack_metric_ui(metric_name: str, metric: "PrivacyRisk", description: str,) -> None:
    """Display privacy metric information.

    Args:
        metric_name (str): Name of the metric (for display).
        metric (PrivacyRisk): Details (e.g., value and confidence interval) for each metric.
        description (str): Basic explanation of the metric for a quick overview.
    """
    st.write(
        f"#### {metric_name} risk:\n"
        f"{description}\n"
    )

    display_privacy_risk_ui(metric_name, metric.value, metric.ci[0], metric.ci[1])


# ============================================================================
# Anonymeter data preparation
# ============================================================================

def value_exists(candidate: object, *dfs: pd.DataFrame) -> bool:
    """Return whether a value occurs anywhere in any supplied dataframe."""
    return any(df.eq(candidate).any().any() for df in dfs)


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
    semantic_type: str,
    dtype: str,
    *dfs: pd.DataFrame,
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
    train: pd.DataFrame,
    test: pd.DataFrame,
    synth: pd.DataFrame,
    stypes: SemanticTypes,
    dtypes: dict[str, str],
) -> AnonymeterInput:
    """Prepare copies of train, test, and synthetic dataframes for Anonymeter evaluation
    in the event that there are missing values. Missing values are temporarily replaced
    with unique markers that are compatible with each column's data type.

    Args:
        train (pd.DataFrame): Training portion of the real dataset.
        test (pd.DataFrame): Test portion of the real dataset.
        synth (pd.DataFrame): Synthetically generated dataset.
        stypes (SemanticTypes): Semantic types assigned to each column.
        dtypes (dict[str, str]): Python storage dtypes for each column.

    Returns:
        AnonymeterInput: Prepared dataframe copies and the missing value markers applied to
        each column.
    """
    train = train.copy()
    test = test.copy()
    synth = synth.copy()

    frames = (train, test, synth)
    markers:dict[str, str | int | float] = {}

    for col, semantic_type in stypes.items():
        if any(df[col].isna().any() for df in frames):
            marker = select_missing_marker(semantic_type, dtypes[col], *frames,)
            markers[col] = marker

            for df in frames:
                df[col] = df[col].fillna(marker)

    for col, dtype in dtypes.items():
        for df in frames:
            if stypes[col] == "Boolean":
                df[col] = df[col].astype(str)

            elif dtype in {"Int64", "int64"}:
                df[col] = pd.to_numeric(df[col]).astype("int64")

            elif dtype == "float64":
                df[col] = df[col].astype("float64")

            elif dtype in {"string", "object"}:
                df[col] = df[col].astype(str)

    return AnonymeterInput(train=train, test=test, synth=synth, missing_markers=markers,)

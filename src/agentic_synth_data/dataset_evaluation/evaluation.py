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

from agentic_synth_data.schemas.attack_model import AttackMetrics, AttackColumns
from agentic_synth_data.schemas.models import AgentEval
from agentic_synth_data.utils import df_formatter_st, format_response, extract_output, parse_json
from agentic_synth_data.prompts.evaluation_prompt import attack_column_selection_message, eval_review_message, evaluation_prompt
from agentic_synth_data.preference_setup import SynthesisRequest
from agentic_synth_data.report import Report


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
    - Jenson-Shannon Distance
    - Normalized Correlation Difference
    - Discriminator AUC

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synthetic_df (pd.DataFrame): The synthetically generated dataframe.

    Returns:
        dict: The decision plan based on the agent's assessment.
    """
    jsd_results = syndat.metrics.jensen_shannon_distance(real=real_df, synthetic=synthetic_df)
    jsd_results["jsd_AVERAGE"] = np.nanmean(list(jsd_results.values())) # In case column to nan

    print("\nJenson-Shannon Distances:")
    for col, jsd_score in jsd_results.items():
        print(col, round(jsd_score, 4))

    norm_corr_diff = syndat.metrics.normalized_correlation_difference(
        real=real_df, synthetic=synthetic_df, ignore_cat=True
        )
    print(f"\nNormalized correlation difference: {round(norm_corr_diff, 4)}")

    discrim_auc = syndat.metrics.discriminator_auc(real=real_df, synthetic=synthetic_df)
    print(f"\nDiscriminator AUC: {round(discrim_auc, 4)}")

    return {
        "jsd": jsd_results,
        "norm_corr_diff": norm_corr_diff,
        "discrim_auc": discrim_auc,
    }


def calculate_metrics_ui(real_df:pd.DataFrame, synthetic_df:pd.DataFrame) -> dict[str, Any]:
    """Calculate metrics comparing the original dataframe (optionally, with missing value markers
     removed) with the synthetic dataframe (UI).

    Calculates SYNDAT metrics:
    - Jenson-Shannon Distance
    - Normalized Correlation Difference
    - Discriminator AUC

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synthetic_df (pd.DataFrame): The synthetically generated dataframe.

    Returns:
        dict: The decision plan based on the agent's assessment.
    """
    st.subheader("Calculate fidelity (syndat) metrics")
    jsd_results = syndat.metrics.jensen_shannon_distance(real=real_df, synthetic=synthetic_df)
    jsd_results["jsd_AVERAGE"] = np.nanmean(list(jsd_results.values())) # In case column to nan

    st.write("#### Jenson-Shannon Distances:")
    jsd_table = pd.DataFrame({
        "Column": jsd_results.keys(),
        "Jenson-Shannon Distance (JSD)": jsd_results.values(),
    })

    df_formatter_st(jsd_table)

    norm_corr_diff = syndat.metrics.normalized_correlation_difference(
        real=real_df, synthetic=synthetic_df, ignore_cat=True,
    )

    st.write("#### Normalized correlation difference:")
    st.write(f"{round(norm_corr_diff, 4)}")

    if len(synthetic_df) < 2:
        st.warning("Discriminator AUC cannot be calculated with fewer than 2 synthetic samples.")
        discrim_auc = None

    else:
        discrim_auc = syndat.metrics.discriminator_auc(real=real_df, synthetic=synthetic_df)
        st.write("#### Discriminator AUC:")
        st.write(f"{round(discrim_auc, 4)}")

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

    return AttackMetrics(
        singling_out= singling_out_risk,
        inference=inference_risk,
        linkability=linkability_risk
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
    st.subheader("Calculate attack metrics")
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

    with st.spinner("Evaluating singling out risk..."):
        singling_out = SinglingOutEvaluator(
            ori=train_df,
            syn=synthetic_df,
            max_attempts=1_000,
            control=test_df,
        )
        singling_out.evaluate()
        singling_out_risk = singling_out.risk()

    st.write(
        f"#### Singling out risk:\n  - Value: {round(singling_out_risk.value*100, 3)}%"
        f"\n  - Confidence interval between: {round(singling_out_risk.ci[0]*100,3)}%"
        f" and {round(singling_out_risk.ci[1]*100,3)}%"
    )

    prompt = attack_column_selection_message(semantic_types=semantic_types)
    st.success(
        "\nAgent is selecting columns for the following privacy metrics to be assessed:"
        "\n- Inference risk\n- Linkability risk"
    )

    raw_response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt,}]}, config=config,
    )

    plan = extract_output(raw_response)

    selected_cols = parse_json(plan, AttackColumns, agent, config,)

    st.write(
        "\nSelected columns:\n- Linkability:\n  - Auxiliary columns: "
        f"{selected_cols.linkability_aux_cols}\n- Inference:\n  - Auxiliary"
        f" columns: {selected_cols.inference_aux_cols}\n  - Secret "
        f"column: {selected_cols.secret}\n\nRationale:"
    )
    st.write(selected_cols.rationale)

    with st.spinner("Evaluating inference risk..."):
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

    st.write(
        f"#### Inference out risk:\n  - Value: {round(inference_risk.value*100, 3)}%"
        f"\n  - Confidence interval: between {round(inference_risk.ci[0]*100,3)}%"
        f" and {round(inference_risk.ci[1]*100,3)}%"
    )

    with st.spinner("Evaluating linkability risk..."):
        linkability = LinkabilityEvaluator(
            ori=train_df,
            syn=synthetic_df,
            aux_cols=selected_cols.linkability_aux_cols,
            n_attacks=min(500, len(train_df), len(test_df), len(synthetic_df),),
            control=test_df,
        )

        linkability.evaluate()
        linkability_risk = linkability.risk()

    st.write(
        f"#### Linkability risk:\n  - Value: {round(linkability_risk.value*100, 3)}%"
        f"\n  - Confidence interval: between {round(linkability_risk.ci[0]*100,3)}%"
        f" and {round(linkability_risk.ci[1]*100,3)}%"
    )

    return AttackMetrics(
        singling_out= singling_out_risk,
        inference=inference_risk,
        linkability=linkability_risk
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
        missing_markers=markers
    )


def generate_plots(real_df, synth_df, version_id):
    """Plot feature distributions and store files."""
    output_dir = Path("../output") / "plots" / f"version_{version_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    syndat.visualization.plot_distributions(real_df, synth_df, store_destination=str(output_dir))

    png_files = sorted(output_dir.glob("*.png"))

    print(f"\nDistribution plot for: {png_files[0].name}")
    display(Image(filename=str(png_files[0])))

    print(f"\n=> Distribution plots for all features saved to: {output_dir}")

    return png_files


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
    prompt = evaluation_prompt(synthesis_request=request_dict, metrics=metrics, attack_metrics=attack_metrics)

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

    with st.spinner("Agent is evaluating the results..."):
        raw_response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt,}]},
            config=config,
        )

        plan = extract_output(raw_response)

        eval_plan = parse_json(plan, AgentEval, agent, config,)

    st.subheader("Evaluation:")
    st.write(f"- Status: {eval_plan.status}")
    if len(eval_plan.problem_columns) > 0:
        st.write(f"- Problem columns: {eval_plan.problem_columns}")
    st.write("- Recommendation:")
    st.write(eval_plan.recommendation)
    st.write("- Rationale:")
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
    output_path = Path("../output") / output_filename
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

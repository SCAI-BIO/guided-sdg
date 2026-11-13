from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from IPython.display import Image, display
import numpy as np
import pandas as pd
import streamlit as st
import syndat

from agentic_synth_data.ui.helpers import CapturedOutput, StreamlitOutput
from agentic_synth_data.schemas.syndat import SyndatMetrics, SyndatOutputs
from agentic_synth_data.utils import OUTPUT_DIR, calculate_delta, df_formatter_st, \
    display_delta_caption


# ============================================================================
# SYNDAT metric calculation
# ============================================================================

def _calculate_metrics(
    real: pd.DataFrame, synth: pd.DataFrame, ui: bool = False,
) -> SyndatMetrics:
    """Calculate and round syndat metrics for evaluation of the synthetic data compared to the
    original dataframe (optionally, with missing value markers removed).

    Calculates metrics:
    - Jensen-Shannon Distance
    - Normalized Correlation Difference
    - Discriminator AUC

    Args:
        real (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synth (pd.DataFrame): The synthetically generated dataframe.
        ui (bool): Whether the metrics are calculated for the UI or CLI. Defaults to False (CLI).

    Returns:
        SyndatMetrics: Calculated metrics and captured output messages.
    """
    if ui:
        jsd_output = StreamlitOutput(live=False)
        norm_corr_diff_output = StreamlitOutput(live=False)
        discrim_auc_output = StreamlitOutput(live=False)

    else:
        jsd_output = CapturedOutput()
        norm_corr_diff_output = CapturedOutput()
        discrim_auc_output = CapturedOutput()

    # Jensen-Shannon Distances:
    with redirect_stdout(jsd_output), redirect_stderr(jsd_output):
        jsd_results = syndat.metrics.jensen_shannon_distance(real=real, synthetic=synth)

    jsd_results["jsd_AVERAGE"] = np.nanmean(list(jsd_results.values())) # In case column to nan

    jsd_results = {
        col: round(float(jsd_score), 4)
        for col, jsd_score in jsd_results.items()
    }

    # Normalized correlation difference:
    with redirect_stdout(norm_corr_diff_output), redirect_stderr(norm_corr_diff_output):
        norm_corr_diff = syndat.metrics.normalized_correlation_difference(
            real=real, synthetic=synth, ignore_cat=True,
        )

    norm_corr_diff = round(norm_corr_diff, 4)

    # Discriminator AUC:
    discrim_auc = None

    if len(synth) >= 2:
        with redirect_stdout(discrim_auc_output), redirect_stderr(discrim_auc_output):
            discrim_auc = syndat.metrics.discriminator_auc(real=real, synthetic=synth,)

        discrim_auc = round(discrim_auc, 4)

    syndat_outputs = SyndatOutputs(
        jsd=jsd_output.buffer or None,
        norm_corr_diff=norm_corr_diff_output.buffer or None,
        discrim_auc=discrim_auc_output.buffer or None,
    )

    return SyndatMetrics(
        jsd=jsd_results,
        norm_corr_diff=norm_corr_diff,
        discrim_auc=discrim_auc,
        output_messages=syndat_outputs,
    )


# ============================================================================
# SYNDAT metric display (CLI and UI)
# ============================================================================

def calculate_metrics(real: pd.DataFrame, synth: pd.DataFrame,) -> SyndatMetrics:
    """Calculate and display syndat metrics.

    Calculates metrics:
    - Jensen-Shannon Distance
    - Normalized Correlation Difference
    - Discriminator AUC

    Args:
        real (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
            missing value markers replaced with nan (norm_na_df).
        synth (pd.DataFrame): The synthetically generated dataframe.

    Returns:
        SyndatMetrics: Calculated metrics and captured output messages.
    """
    metrics = _calculate_metrics(real, synth,)

    # Jensen-Shannon Distances:
    for col, score in metrics.jsd.items():
        print(col, score)

    # Normalized correlation difference:
    print(f"\nNormalized correlation difference: {metrics.norm_corr_diff}")

    # Discriminator AUC:
    if metrics.discrim_auc is None:
        print("Discriminator AUC cannot be calculated with fewer than 2 synthetic samples.")

    else:
        print(f"\nDiscriminator AUC: {metrics.discrim_auc}")

    return metrics


def calculate_metrics_ui(
    real: pd.DataFrame, synth: pd.DataFrame, prev_metrics: SyndatMetrics | None = None,
) -> SyndatMetrics:
    """Calculate and display syndat metrics (UI).

    Calculates metrics:
    - Jensen-Shannon Distance
    - Normalized Correlation Difference
    - Discriminator AUC

    Args:
        real (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
            missing value markers replaced with nan (norm_na_df).
        synth (pd.DataFrame): The synthetically generated dataframe.

    Returns:
        SyndatMetrics: Calculated metrics and captured output messages.
    """
    metrics = _calculate_metrics(real=real, synth=synth, ui=True,)
    display_metrics_ui(metrics=metrics, prev_metrics=prev_metrics,)
    return metrics


def display_metrics_ui(metrics: SyndatMetrics, prev_metrics: SyndatMetrics | None = None,) -> None:
    """Display syndat metrics (UI).

    Args:
        metrics (SyndatMetrics): Calculated metrics and captured output messages.
    """
    st.subheader("Fidelity and utility metrics")
    st.write(
        "The synthetic data should resemble the original dataset. We record the following metrics "
        "from syndat to determine the dataset's fidelity:\n"
        "- Jensen-Shannon Distance\n"
        "- Normalized correlation difference\n"
        "- Discriminator AUC\n\n"
        "All of these metrics output a value between 0 and 1."
    )

    # Jensen-Shannon Distances:
    st.write("#### Jensen-Shannon Distance:")
    st.write(
        "Jensen-Shannon Distance (JSD) quantifies the statistical distributions of each variable "
        "across your real and the synthetic dataset. A value that is closer to 0 indicates that "
        "the statistical distributions identically match the real and synthetic datasets, which is "
        "favorable."
    )

    jsd_table = pd.DataFrame({
        "Column": metrics.jsd.keys(),
        "JSD": metrics.jsd.values(),
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

    prev_jsd = prev_metrics.jsd["jsd_AVERAGE"] if prev_metrics is not None else None

    st.metric(
        label="Jensen-Shannon Divergence (average)",
        value=metrics.jsd["jsd_AVERAGE"],
        delta=calculate_delta(
            current=metrics.jsd["jsd_AVERAGE"],
            previous=prev_jsd,
        ),
        delta_color="inverse",
    )

    display_delta_caption(previous_value=prev_jsd)

    if metrics.output_messages.jsd:
        with st.expander(
            "**UserWarning:** Some variables may have been excluded during computation."
        ):
            st.warning(metrics.output_messages.jsd)

    st.divider()

    # Normalized correlation difference:
    st.write("#### Normalized correlation difference:")
    st.write(
        "Normalized correlation difference calculates the correlation similarity between real and "
        "synthetic datasets and compares their correlation matrices. A value closer to 0 indicates "
        "that the synthetic dataset has greater preservation of the original correlation structure"
        "\n- Note: It is possible for the norm to exceed 1 when one dataset's correlations is "
        "positive and the other negative."
    )

    prev_norm_corr_diff = prev_metrics.norm_corr_diff if prev_metrics is not None else None

    st.metric(
        label="Normalized correlation difference",
        value=metrics.norm_corr_diff,
        delta=calculate_delta(
            current=metrics.norm_corr_diff,
            previous=prev_norm_corr_diff,
        ),
        delta_color="inverse",
    )

    display_delta_caption(previous_value=prev_norm_corr_diff)

    if metrics.output_messages.norm_corr_diff:
        with st.expander(
            "**UserWarning:** Click to view output while calculating normalized correlation "
            "difference."
        ):
            st.warning(metrics.output_messages.norm_corr_diff)

    st.divider()

    # Discriminator AUC:
    if metrics.discrim_auc is None:
        st.warning("Discriminator AUC cannot be calculated with fewer than 2 synthetic samples.")

    else:
        st.write("#### Discriminator AUC:")
        st.write(
            "Discriminator Area Under the Curve (AUC) is a value between 0 and 1. An AUC near 0.5 "
            "is preferable, whereas one near 1 indicates that real and synthetic datasets are easy "
            "to distinguish between."
        )

        prev_discrim_auc = prev_metrics.discrim_auc if prev_metrics is not None else None

        st.metric(
            label="Discriminator AUC",
            value=metrics.discrim_auc,
            delta=calculate_delta(
                current=metrics.discrim_auc,
                previous=prev_discrim_auc,
            ),
            delta_color="inverse",
        )

        display_delta_caption(previous_value=prev_discrim_auc)

        if metrics.output_messages.discrim_auc:
            with st.expander(
                "**UserWarning:** Click to view output while calculating discriminator AUC."
            ):
                st.warning(metrics.output_messages.discrim_auc)


# ============================================================================
# Plot data preparation
# ============================================================================

def encode_for_correlation(
    real: pd.DataFrame, synth: pd.DataFrame, max_categories: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categorical features for correlation analysis while limiting feature expansion.

    Args:
        real (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synth (pd.DataFrame): The synthetically generated dataframe.
        max_categories (int, optional): _description_. Defaults to 10.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    n_real = len(real)
    combined = pd.concat([real, synth], ignore_index=True,)
    combined = combined.loc[:, ~combined.columns.str.startswith("Unnamed:")]

    categorical_cols = combined.select_dtypes(
        include=["object", "string", "category", "bool",]
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


def encode_for_shap(real: pd.DataFrame, synth: pd.DataFrame,) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categorical features numerically for SHAP discrimination analysis.

    Args:
        real (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synth (pd.DataFrame): The synthetically generated dataframe.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    n_real = len(real)
    combined = pd.concat([real, synth], ignore_index=True,)
    combined = combined.loc[:, ~combined.columns.str.startswith("Unnamed:")]

    categorical_cols = combined.select_dtypes(
        include=["object", "string", "category", "bool",]
    ).columns

    for col in categorical_cols:
        combined[col], _ = pd.factorize(combined[col])

    return (combined.iloc[:n_real], combined.iloc[n_real:],)


# ============================================================================
# Plot generation
# ============================================================================

def _generate_plots(
    real: pd.DataFrame, synth: pd.DataFrame, output_dir: Path,
) -> dict[str, list[Path]]:
    """Generate distribution, correlation, and SHAP plots.

    Args:
        real (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                                missing value markers replaced with nan (norm_na_df).
        synth (pd.DataFrame): The synthetically generated dataframe.
        output_dir (Path): Directory for storing generated plots.

    Returns:
        dict[str, list[Path]]: Paths to generated plot images grouped by plot type.
    """
    distribution_dir = output_dir / "distributions"
    correlation_dir = output_dir / "correlations"
    shap_dir = output_dir / "shap"

    for directory in [distribution_dir, correlation_dir, shap_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Distribution plots
    syndat.visualization.plot_distributions(
        real=real,
        synthetic=synth,
        store_destination=str(distribution_dir),
    )

    # Correlation plots
    real_corr, syn_corr = encode_for_correlation(real=real, synth=synth,)
    syndat.visualization.plot_correlations(
        real=real_corr, synthetic=syn_corr, store_destination=str(correlation_dir),
    )

    # SHAP discrimination plot
    real_shap, syn_shap = encode_for_shap(real, synth)
    shap_path = shap_dir / "shap_discrimination.png"

    syndat.visualization.plot_shap_discrimination(
        real=real_shap, synthetic=syn_shap, save_path=str(shap_path),
    )

    return {
        "distributions": sorted(distribution_dir.glob("*.png")),
        "correlations": sorted(correlation_dir.glob("*.png")),
        "shap": sorted(shap_dir.glob("*.png")),
    }



# ============================================================================
# Plot display helpers (UI)
# ============================================================================

def style_status(value: str) -> str:
    """Attribute cell status marker for Jensen-Shannon Distance (JSD) to a specific color. 

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


def _cache_pngs(paths: list[Path]) -> list[dict]:
    """Load PNG files from a directory into memory

    Args:
        paths (Path): Directory containing PNG files.

    Returns:
        list[dict]: Cached PNG data and filenames.
    """
    return [
        {"name": path.name, "data": path.read_bytes()} for path in paths
    ]


@st.fragment
def plot_viewer(images, plot_type: str, version_id: int, show_filename: bool = False,) -> None:
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


# ============================================================================
# Plot generation and display (CLI and UI)
# ============================================================================

def generate_plots(
    real: pd.DataFrame, synth: pd.DataFrame, version_id: int,
) -> dict[str, list[Path]]:
    """Plot feature distributions and store files.

    Args:
        real (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
            missing value markers replaced with nan (norm_na_df).
        synth (pd.DataFrame): The synthetically generated dataframe.
        version_id (int): Denotes the version of the workflow attempt.

    Returns:
        dict[str, list[Path]]: JSD plot images.
    """
    output_dir = OUTPUT_DIR / f"version_{version_id}" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = _generate_plots(
        real=real, synth=synth, output_dir=output_dir,
    )

    for plot_type, paths in plot_paths.items():
        if paths:
            print(f"\n{plot_type.capitalize()} plot: {paths[0].name}")
            display(Image(filename=str(paths[0])))

        print(f"🠊 {plot_type.capitalize()} plots saved to: {output_dir / plot_type}")

    return plot_paths


def generate_plots_ui(
    real: pd.DataFrame, synth: pd.DataFrame, version_id: int,
) -> dict[str, list[Path]]:
    """Generate, cache, store, and display distribution, correlation, and SHAP plots (UI).

    Args:
        real (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
                        missing value markers replaced with nan (norm_na_df).
        synth (pd.DataFrame): The synthetically generated dataframe.
        version_id (int): Denotes the version of the workflow attempt.

    Returns:
        dict[str, list[Path]]: Cached plot images grouped by plot type.
    """
    output_dir = OUTPUT_DIR / f"version_{version_id}" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths_key = f"plot_paths_{version_id}"
    cache_key = f"plots_{version_id}"

    if paths_key not in st.session_state:
        st.session_state[paths_key] = _generate_plots(
            real=real, synth=synth, output_dir=output_dir,
        )

    plot_paths = st.session_state[paths_key]

    if cache_key not in st.session_state:
        st.session_state[cache_key] = {
            plot_type: _cache_pngs(paths) for plot_type, paths in plot_paths.items()
        }

    plots = st.session_state[cache_key]

    tab_dist, tab_corr, tab_shap = st.tabs([
        "Distribution plots",
        "Correlation plots",
        "SHAP discrimination plot",
    ])

    with tab_dist:
        st.write(
            "Compare the distribution of individual variables between the real and synthetic "
            "datasets to assess how well their statistical properties are preserved."
        )

        plot_viewer(
            images=plots["distributions"],
            plot_type="distribution",
            version_id=version_id,
        )

    with tab_corr:
        st.write(
            "Compare relationships between variables in the real and synthetic datasets to "
            "assess how well their correlation structures are preserved."
        )

        plot_viewer(
            images=plots["correlations"],
            plot_type="correlation",
            version_id=version_id,
            show_filename=True,
        )

    with tab_shap:
        st.write(
            "Identify which variables contribute most to a classifier's ability to distinguish "
            "real data from synthetic data."
        )

        plot_viewer(
            images=plots["shap"],
            plot_type="shap",
            version_id=version_id,
        )

    return plot_paths

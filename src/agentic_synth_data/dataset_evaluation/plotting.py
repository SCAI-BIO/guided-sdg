import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from agentic_synth_data.workflow import DataGen


def compile_metrics(versions: list[DataGen]) -> pd.DataFrame:
    """Compile syndat and anonymeter metrics across versions for display.

    Args:
        versions (list[DataGen]): Existing versions.

    Returns:
        pd.DataFrame: Dataframe containing version, category, metric and score.
    """
    records = []

    for gen in versions:
        syndat = gen.report.evaluate.syndat
        attack = gen.report.evaluate.attack

        if syndat is None or attack is None:
            continue

        metrics = {
            "syndat": {
                "Average JSD": syndat.jsd["jsd_AVERAGE"],
                "Normalized correlation difference": syndat.norm_corr_diff,
                "Discriminator AUC": syndat.discrim_auc,
            },
            "anonymeter": {
                "Singling out risk": attack.singling_out.value,
                "Linkability risk": attack.linkability.value,
                "Inference risk": attack.inference.value,
            },
        }

        for category, category_metrics in metrics.items():
            for metric, score in category_metrics.items():
                records.append({
                    "version": gen.version_id, "category": category, "metric": metric,
                    "score": score,
                })

    return pd.DataFrame(records, columns=["version", "category", "metric", "score"])


def summarize_tradeoff(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the fidelity–privacy trade-off for each workflow version.

    The trade-off is calculated as the mean of available normalized Syndat fidelity scores
    minus the mean of available Anonymeter privacy risks. Missing scores are excluded from
    the averages.

    Args:
        df (pd.DataFrame): Dataframe with version, category, metric, and score.

    Returns:
        pd.DataFrame: Dataframe with version, mean Syndat score, mean Anonymeter risk, and
            calculated trade-off.
    """
    summary = df.groupby(["version", "category"])["score"].mean().unstack()

    # Both fidelity and privacy must have at least one valid metric
    summary = summary.dropna(subset=["syndat", "anonymeter"])

    summary["tradeoff"] = summary["syndat"] - summary["anonymeter"]

    return summary.reset_index()


def plot_metrics(df: pd.DataFrame) -> Figure:
    """Plot normalized fidelity, privacy risk, and trade-off scores across workflow versions.
    The trade-off represents mean fidelity minus mean privacy risk.

    Args:
        df (pd.DataFrame): Dataframe containing version, category, metric and score.

    Returns:
        Figure: Matplotlib figure comparing individual metrics and the
            fidelity–privacy trade-off across workflow versions.
    """
    df = df.copy()

    metric_groups = {
        "syndat": ["Average JSD", "Normalized correlation difference", "Discriminator AUC"],
        "anonymeter": ["Singling out risk", "Linkability risk", "Inference risk"],
    }

    colors = {"syndat": "blue", "anonymeter": "orange"}

    markers = {
        metric: marker for metrics in metric_groups.values()
        for metric, marker in zip(metrics, ["o", "s", "^"])
    }

    # Visually separate overlapping points
    df["x"] = df["version"].astype(float)
    for _, indices in df.groupby(["version", df["score"].round(3)]).groups.items():
        if len(indices) > 1:
            df.loc[indices, "x"] += np.linspace(-0.018, 0.018, len(indices))

    fig, ax = plt.subplots(figsize=(9, 5))

    # Connect individual metric scores across versions
    for (category, _), group in df.groupby(["category", "metric"]):
        group = group.sort_values("version")

        ax.plot(group["x"], group["score"], color=colors[category], alpha=0.6,)

    sns.scatterplot(
        data=df, x="x", y="score", hue="category", style="metric", palette=colors, markers=markers,
        s=35, edgecolor="black", linewidth=0.5, legend=False, ax=ax, zorder=2,
    )

    # Plot trade-off line
    tradeoff_df = summarize_tradeoff(df)

    ax.plot(
        tradeoff_df["version"], tradeoff_df["tradeoff"], color="green", marker="D",
        markersize=np.sqrt(35), markeredgecolor="black", markeredgewidth=0.5, linewidth=2,
        zorder=3,
    )

    ax.set_xlabel("Version")
    ax.set_ylabel("Normalized score / trade-off (points)")
    ax.set_xticks(sorted(df["version"].unique()))
    ax.set_xlim(df["version"].min() - 0.5, df["version"].max() + 0.5)

    # Group metrics in the legend
    handles = []
    for category, metrics in metric_groups.items():
        if handles:
            handles.append(Line2D([], [], linestyle="None", label=" "))

        handles.append(Line2D([], [], linestyle="None", label=category))

        for metric in metrics:
            handles.append(
                Line2D(
                    [], [], marker=markers[metric], linestyle="None", color=colors[category],
                    markeredgecolor="black", label=f"   {metric}",
                )
            )

    # Add trade-off line to the legend
    handles.append(Line2D([], [], linestyle="None", label= " "))
    handles.append(
        Line2D(
            [], [], color="green", marker="D", markeredgecolor="black", linewidth=2,
            label="Fidelity privacy trade-off",
        )
    )

    legend = ax.legend(
        handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
        handlelength=1.2,
    )

    for label in legend.get_texts():
        if label.get_text() in metric_groups:
            label.set_weight("bold")
            label.set_text(label.get_text().capitalize())

    fig.tight_layout()
    return fig

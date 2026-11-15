from io import BytesIO
from pathlib import Path
from matplotlib import pyplot as plt
import streamlit as st

from agentic_synth_data.revision import version_tabs_ui
from agentic_synth_data.evaluation.plot import compile_metrics, plot_metrics
from agentic_synth_data.report import display_report_ui
from agentic_synth_data.utils import display_downloads_ui, display_workflow_progress


st.sidebar.title("GuidedSDG")

display_workflow_progress(
    "`✓ Preferences ─ ✓ Data types ─ ✓ Generator ─ ✓ Missing values ─ ✓ Generate ─ ✓ Evaluate`"
)

st.header("Workflow reports")
st.write("Review SDG workflow decisions and results.")
st.space("small")

# No workflow has been started.
if not st.session_state.get("revision_history"):
    st.info(
        "No workflow has been started yet. Complete an initial run through steps to generate a "
        "report."
    )

    st.stop()

revision_tabs, comparison_tab = version_tabs_ui(
    label="Version", key="active_report_tab", include_initial=True, compare=True,
)

for version_id, tab in revision_tabs.items():
    if not tab.open:
        continue

    with tab:
        revision = st.session_state.revision_history.get(version_id)

        if revision is None:
            st.warning(f"Version {version_id} was not found.")
            continue

        display_report_ui(revision)

        display_downloads_ui(version_id)

if comparison_tab is not None and comparison_tab.open:
    with comparison_tab:
        st.subheader("Compare workflow versions")
        st.write("Compare evaluation metrics across workflow versions.")

        versions = sorted(
            st.session_state.get("revision_history", {}).values(), key=lambda gen: gen.version_id,
        )

        data_name = Path(versions[0].data_path).stem

        metrics_df = compile_metrics(versions=versions)

        if metrics_df.empty:
            st.info("No evaluation metrics available")

        else:
            version_list = list(st.session_state.revision_history)
            st.dataframe(metrics_df, hide_index=True, width="stretch",) # NOTE: Temp view...

            selected_versions = st.multiselect(
                label="Select versions to compare",
                options=version_list,
                default=version_list,
                key="selected_versions_to_view",
            )

            filtered_df = metrics_df[metrics_df["version"].isin(selected_versions)]

            if not filtered_df.empty:
                missing = filtered_df[filtered_df["score"].isna()]

                if not missing.empty:
                    MISSING_DETAILS = "\n".join(
                        f"Version {row.version}: {row.metric}" for row in missing.itertuples()
                    )

                    st.warning(
                        "Some metrics were unavailable and were excluded from the trade-off "
                        f"calculation:\n\n{MISSING_DETAILS}\n\nSee the corresponding version "
                        "tabs for additional details."
                    )

                fig = plot_metrics(filtered_df)
                st.pyplot(fig)

                st.caption(
                    "Comparison of normalized fidelity (Syndat; higher is better) and privacy risk "
                    "(Anonymeter; lower is better) scores across versions, scaled to 0–100. The "
                    "green line represents the fidelity and privacy trade-off "
                    "`(mean fidelity - mean privacy risk)`, with available metrics equally "
                    "weighted."
                )

                buffer = BytesIO()
                fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")

                VERSION_SUFFIX = "_".join(f"v{v}" for v in sorted(selected_versions))

                _, center, _ = st.columns(3)
                with center:
                    st.download_button(
                        label="Download plot",
                        data=buffer.getvalue(),
                        file_name=f"{data_name}_comparison_{VERSION_SUFFIX}.png",
                        mime="image/png",
                        on_click="ignore",
                    )

                plt.close(fig)

            else:
                st.info("Select at least one version to display the figure.")

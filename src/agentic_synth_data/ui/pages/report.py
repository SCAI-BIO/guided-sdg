import streamlit as st

from agentic_synth_data.revision import version_tabs_ui
from agentic_synth_data.dataset_evaluation.plotting import compile_metrics, plot_metrics
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

        metrics_df = compile_metrics(versions=versions)

        if metrics_df.empty:
            st.info("No evaluation metrics available")

        else:
            version_list = list(st.session_state.revision_history)
            st.dataframe(metrics_df, hide_index=True, width="stretch",)

            # NOTE: possibly change message to "deselect versions to compare"
            st.multiselect(
                label="Select versions to compare",
                options=version_list,
                default=version_list,
                key="selected_versions_to_view",
            )

            st.pyplot(plot_metrics(metrics_df)) # NOTE: possibly plotly?
            # NOTE: possibly also option for downloading?

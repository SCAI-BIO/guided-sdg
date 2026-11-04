import streamlit as st

from agentic_synth_data.rerun_workflow import version_tabs_ui
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

# TODO: Include report comparison as well.

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

        display_downloads_ui(version_id)

if comparison_tab is not None and comparison_tab.open:
    with comparison_tab:
        st.subheader("Compare versions")
        st.write("Compare evaluation metrics across workflow versions.")

        st.multiselect(
            "Select versions to compare", options=list(st.session_state.revision_history)
        )

    # TODO: Add a clearer method on how to compare reports
        # This or button:
            # st.checkbox("Are these correct?")

import streamlit as st

from agentic_synth_data.rerun_workflow import version_tabs_ui
from agentic_synth_data.utils import display_workflow_progress, page_navigation
from agentic_synth_data.dataset_evaluation.syndat_metrics import display_metrics_ui
from agentic_synth_data.dataset_evaluation.attack_metrics import ANONYMETER_METRIC_DESCRIPTION, \
    display_attack_metrics_ui


st.sidebar.title("GuidedSDG")

display_workflow_progress(
    "`✓ Preferences ─ ✓ Data types ─ ✓ Generator ─ ✓ Missing values ─ ✓ Generate ─ ✓ Evaluate`"
)

st.header("Consider workflow revision")
st.space("small")

st.write(
    "All main workflow steps have been completed, but now, you determine whether you would like to "
    "try alternative choices in the workflow or not. If the results are satisfactory to you, "
    "'finish' can be selected. Otherwise, select 'revise' to revise any of the following decisions:"
    "\n- Generator selection choice.\n"
    "- Parameter selection choice.\n"
    "- Missing value handling decision."
)

# Initialize revision history
if "revision_history" not in st.session_state:
    st.session_state.revision_history = {1: st.session_state.gen,}

initial = st.session_state.revision_history[1]
gen = st.session_state.get("current_revision", initial,)

with st.expander(
    "Click to review fidelity, utility, and privacy metrics from the initial version."
):
    col1, col2 = st.columns(2)
    with col1:
        display_metrics_ui(metrics=initial.report.evaluate.syndat,)

    with col2:
        display_attack_metrics_ui(
            attack_metrics=initial.report.evaluate.attack,
            description=ANONYMETER_METRIC_DESCRIPTION,
            key_prefix="initial_review",
        )

pending_tab = st.session_state.pop("pending_active_revision_tab", None)

if pending_tab is not None:
    st.session_state.active_revision_tab = pending_tab

revision_tabs = version_tabs_ui()

# Original workflow (prior to revisions existing)
if not revision_tabs:
    gen.post_evaluation_check_ui()
    st.stop()

for version_id, tab in revision_tabs.items():
    if not tab.open:
        continue

    with tab:
        pending_from = st.session_state.get("revision_pending_from")

        # User is selecting changes for a new revision
        if pending_from is not None and version_id == pending_from + 1:
            parent = st.session_state.revision_history[pending_from]

            parent.post_evaluation_check_ui()
            continue

        # Retrieve the revision
        revision = st.session_state.revision_history.get(version_id)

        if revision is None:
            st.error(f"Revision {version_id - 1} was not found.")
            continue

        # Execute pending revision
        if revision.pending_revision_plan is not None:
            revision._rerun_selection_ui(revision.pending_revision_plan)
            revision.pending_revision_plan = None

            st.rerun()

        revision.display_revision_summary_ui()
        revision.display_revision_results_ui()

        if version_id == gen.version_id:
            revision.post_evaluation_check_ui()

page_navigation(
    back_page="pages/evaluate.py",
    next_page="pages/missing_handling.py",
    next_disabled=True,
)

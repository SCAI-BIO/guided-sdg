import streamlit as st

from agentic_synth_data.rerun_workflow import revision_tabs_ui
from agentic_synth_data.utils import page_navigation


st.sidebar.title("GuidedSDG")

st.write(
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

gen = st.session_state.get("current_revision", st.session_state.gen)

if "revision_history" not in st.session_state:
    st.session_state.revision_history = {}

st.session_state.revision_history[1] = st.session_state.gen

############################
st.write("Current version:", gen.version_id)
st.write(
    "Stored revision:",
    st.session_state.current_revision.version_id
    if "current_revision" in st.session_state
    else None,
)
st.write("Has pending plan:", gen.pending_revision_plan is not None)
############################

pending_tab = st.session_state.pop("pending_active_revision_tab", None)

if pending_tab is not None:
    st.session_state.active_revision_tab = pending_tab

revision_tabs = revision_tabs_ui()

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

            if version_id == gen.version_id:
                revision.post_evaluation_check_ui()

        # TODO: Display completed revision
        else:
            if version_id == gen.version_id:
                revision.post_evaluation_check_ui()

page_navigation(
    back_page="pages/evaluate.py",
    next_page="pages/missing_handling.py",
    next_disabled=True,
)

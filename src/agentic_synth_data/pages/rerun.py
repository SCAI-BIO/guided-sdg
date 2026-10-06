import streamlit as st

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

# Evaluation
gen = st.session_state.gen
gen.post_evaluation_check_ui()

page_navigation(back_page="pages/synthetic_generation.py", next_page="pages/missing_handling.py", next_disabled=True,)
# NOTE: next page is inaccurate for now, but is unused/disabled

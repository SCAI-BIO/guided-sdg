import streamlit as st

from agentic_synth_data.utils import display_workflow_progress, page_navigation


st.sidebar.title("GuidedSDG")

display_workflow_progress(
    "`✓ Preferences ─ ● Data types ─ ○ Generator ─ ○ Missing values ─ ○ Generate ─ ○ Evaluate`"
)

st.header("Prepare dataset: Detect data types")
st.space("small")

st.write(
    "To generate synthetic data, the data generators need to know the data types for each "
    "variable (column). Data types should be recorded appropriately to avoid errors or "
    "mistakes that can affect the workflow and make SDG not possible with the dataset."
)

# Handle dtypes
gen = st.session_state.gen
request = gen.preprocess_dtypes_ui()

page_navigation(back_page="pages/preferences.py", next_page="pages/select.py")

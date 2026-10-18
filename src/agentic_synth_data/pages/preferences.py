import streamlit as st

from agentic_synth_data.utils import page_navigation


st.sidebar.title("GuidedSDG")

st.write(
    "`● Preferences ─ ○ Data types ─ ○ Generator ─ ○ Missing values ─ ○ Generate ─ ○ Evaluate`"
)

st.header("Select preferences")
st.space("small")

st.write(
    "Your preferences help guide data preparation and synthetic data generation based on your "
    "priorities and requirements. These initial preferences serve as a starting point and can be "
    "reviewed and modified throughout the workflow."
)

# Render preference settings
gen = st.session_state.gen
request = gen.apply_preferences_ui()

if request is not None:
    st.session_state.gen.synthesis_request = request

page_navigation(
    back_page="pages/setup.py", next_page="pages/handle_dtypes.py", next_disabled=request is None,
)

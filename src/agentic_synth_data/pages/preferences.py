import streamlit as st

from agentic_synth_data.utils import page_navigation


st.sidebar.title("GuidedSDG")

st.write(
    "`● Preferences ─ ○ Data types ─ ○ Generator ─ ○ Missing values ─ ○ Generate ─ ○ Evaluate`"
)

st.header("Select preferences")
st.space("small")

st.write(
    "Your initial preprocessing and generation preferences are important as they can be used to "
    "indicate how to best proceed in generating synthetic data. This is just a starting point and "
    "your preferences will be updated throughout the workflow if you decide to change anything."
)

# Render preference settings
gen = st.session_state.gen
request = gen.apply_preferences_ui()

if request is not None:
    st.session_state.gen.synthesis_request = request

page_navigation(back_page="pages/setup.py", next_page="pages/handle_dtypes.py", next_disabled=request is None,)

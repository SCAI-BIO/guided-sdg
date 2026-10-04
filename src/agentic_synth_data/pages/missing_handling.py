import streamlit as st

from agentic_synth_data.utils import page_navigation


st.sidebar.title("GuidedSDG")

st.write(
    "`✓ Preferences ─ ✓ Data types ─ ✓ Generator ─ ● Missing values ─ ○ Generate ─ ○ Evaluate`"
)

st.header("Prepare dataset: Missing value handling")
st.space("small")

st.write(
    "Some data generators do not allow missing values in the dataset. If the dataset has "
    "missing values in those cases, it can result in an error. Regardless of whether you have "
    "selected a specific strategy to handle missing values or indicated you wanted a suggestion "
    "from the LLM agent, you choose how to proceed."
)

# Missing handling
gen = st.session_state.gen
request = gen.preprocess_missing_vals_ui()

page_navigation(back_page="pages/select.py", next_page="pages/generate.py")

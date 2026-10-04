import streamlit as st

from agentic_synth_data.utils import page_navigation


st.sidebar.title("GuidedSDG")

st.write(
    "`✓ Preferences ─ ✓ Data types ─ ● Generator ─ ○ Missing values ─ ○ Generate ─ ○ Evaluate`"
)

st.header("Prepare dataset: Generator selection")
st.space("small")

st.write(
    "Different data generators may create better synthetic data with particular datasets, "
    "depending on their characteristics. Additionally, different data generators may have a "
    "greater focus on privacy preservation than others. In this step, the summary statistics and "
    "correlations of the dataset are recorded to inform the LLM agent about the dataset. Whether "
    "the data generator has been selected or a suggestion has been requested, the agent will "
    "assess the chosen data generator to determine if it has a greater likelihood of producing "
    "usable data that has high fidelity, utility, and preserves privacy."
)

# Generator selection
gen = st.session_state.gen
request = gen.generator_selection_ui()

page_navigation(back_page="pages/handle_dtypes.py", next_page="pages/missing_handling.py",)

# TODO: improve visibility
# Generator selection (preparation setup)
# Display summary statistics (table)
# Heatmap (colorblind friendly)
# Agent is generating a recommendation (i) [hover info bar]

#privacy = st.selectbox("Privacy", ["s", "d"], help ="exlpland")
# Recommendation: generator name (i) brief information on the generator
# Accept\revisit\ask
# back\revisit

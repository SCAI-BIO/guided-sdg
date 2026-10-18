import streamlit as st

from agentic_synth_data.utils import page_navigation


st.sidebar.title("GuidedSDG")

st.write(
    "`✓ Preferences ─ ✓ Data types ─ ● Generator ─ ○ Missing values ─ ○ Generate ─ ○ Evaluate`"
)

st.header("Prepare dataset: Generator selection")
st.space("small")

st.write(
    "Synthetic data generators differ in their ability to preserve data fidelity, utility, and "
    "privacy, depending on the characteristics of the original dataset. In this step, dataset "
    "summary statistics and correlations are collected to inform the LLM agent's assessment. Based "
    "on these characteristics and your preferences, the agent can recommend a suitable generator "
    "or evaluate your selected generator. You can then review and revise the recommendation "
    "before proceeding."
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

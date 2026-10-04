import streamlit as st

from agentic_synth_data.utils import page_navigation


st.sidebar.title("GuidedSDG")

st.write(
    "`✓ Preferences ─ ✓ Data types ─ ✓ Generator ─ ✓ Missing values ─ ✓ Generate ─ ● Evaluate`"
)

st.header("Evaluation")
st.space("small")

st.write("Evaluate whether the synthetic data is satisfactory.")
st.write(
    "This step involves combining metrics from both [syndat](https://syndat.scai.fraunhofer.de/) "
    "and from [anonymeter](https://github.com/statice/anonymeter) to assess the fidelity and "
    "utility as well as the privacy of the synthetic dataset."
)

# Evaluation
gen = st.session_state.gen

with st.spinner("Running evaluation..."):
    request = gen.evaluate_ui()

page_navigation(back_page="pages/generate.py", next_page="pages/rerun.py",)

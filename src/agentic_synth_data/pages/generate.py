import streamlit as st

from agentic_synth_data.utils import page_navigation


st.sidebar.title("GuidedSDG")

st.write(
    "`✓ Preferences ─ ✓ Data types ─ ✓ Generator ─ ✓ Missing values ─ ● Generate ─ ○ Evaluate`"
)

st.header("Generation")
st.space("small")

st.write(
    "Now that data types, the generator, and missing value handling strategy have been determined "
    "it is time for SDG. Metadata is created for the data generator selected and now the data "
    "generator is running."
)

# Generator selection
gen = st.session_state.gen

st.subheader("Create metadata for tool use") # TODO changed
gen.save_prepared_data_ui()
gen.create_metadata()

st.subheader("Synthesize synthetic data")
gen.run_synthesis_ui()

page_navigation(back_page="pages/missing_handling.py", next_page="pages/evaluate.py")

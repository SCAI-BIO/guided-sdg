import streamlit as st


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

def go_next():
    st.switch_page("pages/evaluate.py")

def go_back():
    # TODO: Add a warning (are you sure you want to go back to x step?)
    st.switch_page("pages/missing_handling.py")

col1, col2 = st.columns(2)
with col1:
    back = st.button("back", on_click=go_back, key="generate_back",)
    # add an " are you sure you want to go back button"

with col2:
    # make continue pop up once everything is done
    cont = st.button("continue", on_click=go_next, key="generate_cont",)

import streamlit as st


st.sidebar.title("Agentic synthetic data workflow")

st.write(
    "✓ Preferences ─ ✓ Data types ─ ✓ Generator ─ "
    "✓ Missing values ─ ● Generate ─ ○ Evaluate"
)
st.write("---")
st.header("Generation")
st.space("small")

# Generator selection
gen = st.session_state.gen
gen.save_prepared_data_ui()
gen.create_metadata()
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

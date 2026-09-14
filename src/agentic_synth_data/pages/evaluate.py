import streamlit as st


st.sidebar.title("Agentic synthetic data workflow")

st.write(
    "✓ Preferences ─ ✓ Data types ─ ✓ Generator ─ "
    "✓ Missing values ─ ✓ Generate ─ ● Evaluate"
)

st.write("---")
st.header("Evaluation")
st.space("small")

# Evaluation
gen = st.session_state.gen
request = gen.evaluate_ui()

def go_next():
    st.switch_page("pages/rerun.py")

def go_back():
    # TODO: Add a warning (are you sure you want to go back to x step?)
    st.switch_page("pages/synthetic_generation.py")

col1, col2 = st.columns(2)
with col1:
    back = st.button("back", on_click=go_back, key="evaluate_back",)
    # add an " are you sure you want to go back button"

with col2:
    # make continue pop up once everything is done
    cont = st.button("continue", on_click=go_next, key="evaluate_cont",)

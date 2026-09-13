import streamlit as st


st.sidebar.title("Agentic synthetic data workflow")

st.write(
    "✓ Preferences ─ ● Data types ─ ○ Generator ─ "
    "○ Missing values ─ ○ Generate ─ ○ Evaluate"
)
st.write("---")

st.header("Prepare dataset: Detect data types")
st.space("small")

# Handle dtypes
gen = st.session_state.gen
request = gen.preprocess_dtypes_ui()

col1, col2 = st.columns(2)
with col1:
    back = st.button("back")
    # add an " are you sure you want to go back button"

with col2:
    # make continue pop up once everything is done
    cont = st.button("continue",) # disabled=x is None)

if cont:
    st.switch_page("pages/select.py")

if back:
    # TODO: Add a warning (are you sure you want to go back to x step?)
    st.switch_page("pages/preferences.py")

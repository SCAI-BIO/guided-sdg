import streamlit as st


st.sidebar.title("Agentic synthetic data workflow")

st.write(
    "● Preferences ─ ○ Data types ─ ○ Generator ─ "
    "○ Missing values ─ ○ Generate ─ ○ Evaluate"
)
st.write("---")

# Render preference settings
gen = st.session_state.gen
request = gen.apply_preferences_ui()

col1, col2 = st.columns(2)
with col1:
    back = st.button("back")
    # TODO: add an " are you sure you want to go back button"

with col2:
    # TODO: make continue pop up once everything is done
    cont = st.button("continue")

if cont:
    if request is None:
        st.warning("Please complete all preference selections before continuing.")
    else:
        st.session_state.gen.synthesis_request = request
        st.switch_page("pages/handle_dtypes.py")

if back:
    # TODO: Add a warning (are you sure you want to go back to x step?)
    st.switch_page("pages/setup.py")

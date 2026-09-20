import streamlit as st


st.sidebar.title("GuidedSDG")

st.write(
    "`✓ Preferences ─ ✓ Data types ─ ✓ Generator ─ ● Missing values ─ ○ Generate ─ ○ Evaluate`"
)

st.header("Prepare dataset: Missing value handling")
st.space("small")

st.write(
    "Some data generators do not allow missing values in the dataset. If the dataset has "
    "are missing values in those cases, it can result in an error. Regardless of whether you have "
    "selected a specific strategy to handle missing values or indicated you wanted a suggestion "
    "from the LLM agent, you choose how to proceed."
)

# Missing handling
gen = st.session_state.gen
request = gen.preprocess_missing_vals_ui()

def go_next():
    st.switch_page("pages/generate.py")

def go_back():
    # TODO: Add a warning (are you sure you want to go back to x step?)
    st.switch_page("pages/select.py")

col1, col2 = st.columns(2)
with col1:
    back = st.button("back", on_click=go_back, key="missing_handling_back",)
    # add an " are you sure you want to go back button"

with col2:
    # make continue pop up once everything is done
    cont = st.button("continue", on_click=go_next, key="missing_handling_cont",)

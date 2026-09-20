import streamlit as st


st.sidebar.title("GuidedSDG")

st.write(
    "`✓ Preferences ─ ● Data types ─ ○ Generator ─ ○ Missing values ─ ○ Generate ─ ○ Evaluate`"
)

st.header("Prepare dataset: Detect data types")
st.space("small")

st.write(
    "To generate synthetic data, the data generators need to know the data types for each "
    "variable (column). This step involves three parts:\n"
    "1. Look for any missing value markers that should be encoded as null values within the "
    "dataset prior to analysis.\n2. Automatically detect the majority Python data types present "
    "for each variable (column) in the dataframe.\n3. Have the agent determine the semantic "
    "meaning of each variable (column).\nData types should be recorded appropriately to avoid "
    "errors or mistakes that can affect the workflow and make SDG not possible with the dataset."
)

# Handle dtypes
gen = st.session_state.gen
request = gen.preprocess_dtypes_ui()

def go_next():
    st.switch_page("pages/select.py")

def go_back():
    # TODO: Add a warning (are you sure you want to go back to x step?)
    st.switch_page("pages/preferences.py")

col1, col2 = st.columns(2)
with col1:
    back = st.button("back", on_click=go_back, key="dtype_back",)
    # add an " are you sure you want to go back button"

with col2:
    # make continue pop up once everything is done
    cont = st.button("continue", on_click=go_next, key="dtype_cont",)

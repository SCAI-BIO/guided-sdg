import streamlit as st


st.sidebar.title("GuidedSDG")

st.write(
    "`● Preferences ─ ○ Data types ─ ○ Generator ─ ○ Missing values ─ ○ Generate ─ ○ Evaluate`"
)

st.header("Select preferences")
st.space("small")

st.write(
    "Your initial preprocessing and generation preferences are important as they can be used to "
    "indicate how to best proceed in generating synthetic data. This is just a starting point and "
    "your preferences will be updated throughout the workflow if you decide to change anything."
)

# Render preference settings
gen = st.session_state.gen
request = gen.apply_preferences_ui()

def go_next():
    if request is None:
        st.warning("Please complete all preference selections before continuing.")
    else:
        st.session_state.gen.synthesis_request = request
        st.switch_page("pages/handle_dtypes.py")

def go_back():
    # TODO: Add a warning (are you sure you want to go back to x step?)
    st.switch_page("pages/setup.py")

col1, col2 = st.columns(2)
with col1:
    back = st.button("back", on_click=go_back, key="preferences_back",)
    # add an " are you sure you want to go back button"

with col2:
    # make continue pop up once everything is done
    cont = st.button("continue", on_click=go_next, key="preferences_cont",)

import streamlit as st


st.sidebar.title("GuidedSDG")

st.write(
    "`✓ Preferences ─ ✓ Data types ─ ✓ Generator ─ ✓ Missing values ─ ✓ Generate ─ ✓ Evaluate`"
)

st.header("Consider workflow revision")
st.space("small")

st.write(
    "All main workflow steps have been completed, but now, you determine whether you would like to "
    "try alternative choices in the workflow or not. If the results are satisfactory to you, "
    "'finish' can be selected. Otherwise, select 'revise' to revise any of the following decisions:"
    "\n- Generator selection choice.\n- Parameter selection choice.\n- Missing value handling "
    "decision."
)

# Evaluation
gen = st.session_state.gen
gen.post_evaluation_check_ui()

def go_next():
    #st.switch_page("pages/missing_handling.py")
    st.write("Check to rerun the workflow")
    st.write("currently not implemented")

def go_back():
    # TODO: Add a warning (are you sure you want to go back to x step?)
    st.switch_page("pages/synthetic_generation.py")

col1, col2 = st.columns(2)
with col1:
    back = st.button("back", on_click=go_back, key="rerun_back",)
    # add an " are you sure you want to go back button"

with col2:
    # make continue pop up once everything is done
    cont = st.button("continue", on_click=go_next, key="rerun_cont",)

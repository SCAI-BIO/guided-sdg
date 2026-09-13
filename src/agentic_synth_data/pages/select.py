import streamlit as st


st.sidebar.title("Agentic synthetic data workflow")

st.write(
    "✓ Preferences ─ ✓ Data types ─ ● Generator ─ "
    "○ Missing values ─ ○ Generate ─ ○ Evaluate"
)
st.write("---")

# Generator selection
gen = st.session_state.gen
request = gen.generator_selection_ui()

# Will need to write the print statements...
col1, col2 = st.columns(2)
with col1:
    back = st.button("back")
    # add an " are you sure you want to go back button"

with col2:
    # make continue pop up once everything is done
    cont = st.button("continue")

if cont:
    st.switch_page("./pages/missing_handling.py")

if back:
    # TODO: Add a warning (are you sure you want to go back to x step?)
    st.switch_page("./pages/handle_dtypes.py")

# TODO: improve visibility
# Generator selection (preparation setup)
# Display summary statistics (table)
# Heatmap (colorblind friendly)
# Agent is generating a recommendation (i) [hover info bar]

#privacy = st.selectbox("Privacy", ["s", "d"], help ="exlpland")
# Recommendation: generator name (i) brief information on the generator
# Accept\revisit\ask
# back\revisit

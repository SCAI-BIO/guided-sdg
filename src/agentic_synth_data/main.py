from dotenv import load_dotenv
import streamlit as st

load_dotenv()

reports = st.Page("./pages/report.py", title="Reports") # Hidden (for now)
setup = st.Page("./pages/setup.py", title="0. Setup")
select_preferences = st.Page("./pages/select_preferences.py", title="1. Select preferences")
handle_dtypes = st.Page("./pages/handle_dtypes.py", title="2. Handle data types")
generator_selection = st.Page("./pages/generator_selection.py", title="3. Generator selection")
missing_handling = st.Page("./pages/missing_value_handling.py", title="4. Missing value handling")
generation = st.Page("./pages/synthetic_generation.py", title="5. Synthetic generation")
evaluation = st.Page("./pages/evaluation.py", title="6. Evaluation")

st.navigation(
    [
        setup,
        select_preferences,
        handle_dtypes,
        generator_selection,
        missing_handling,
        generation,
        evaluation,
    ]
).run()

# TODO add some sort of home button? Explains the project on a base level (and features?)
# Add a workflow button (allows user to return to workflow)

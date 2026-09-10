from dotenv import load_dotenv
import streamlit as st

load_dotenv()

reports = st.Page("./pages/report.py", title="Reports")
setup = st.Page("./pages/setup.py", title="0. Setup")
select_preferences = st.Page("./pages/preferences.py", title="1. Select preferences")
handle_dtypes = st.Page("./pages/handle_dtypes.py", title="2. Handle data types")
generator_selection = st.Page("./pages/select.py", title="3. Generator selection")
missing_handling = st.Page("./pages/missing_handling.py", title="4. Missing value handling")
generation = st.Page("./pages/generate.py", title="5. Synthetic generation")
evaluation = st.Page("./pages/evaluate.py", title="6. Evaluation")

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

# TODO Add a homepage (explains the project and its features)
# TODO Add a workflow page (includes all steps besides setup which are currently present in navigation)

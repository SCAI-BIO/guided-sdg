from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from agentic_synth_data.utils import colorize_multiselect


IMAGES_DIR = Path(__file__).resolve().parents[1] / "images"

LOGO_PATH = Path("/src/agentic_synth_data/images/icon.jpg")


load_dotenv()

about_msg = (
    "GuidedSDG is an agentic decision-support workflow for Synthetic Data Generation (SDG). A "
    "Large Language Model (LLM) agent assists users in preparing datasets, selecting suitable "
    "generators, and evaluating synthetic data based on the characteristics of the original "
    "dataset and user preferences."
)

st.set_page_config(
    page_title="GuidedSDG",
    page_icon=str(IMAGES_DIR / "analysis.png"),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/SCAI-BIO/agentic-synth-data/tree/feat29/user-interface",
        "Report a bug": "https://github.com/SCAI-BIO/agentic-synth-data/issues",
        "About": about_msg,
    },
)

st.logo(
    IMAGES_DIR / "logo-img.png",
    icon_image=IMAGES_DIR / "logo-img.png",
    size="large",
)

colorize_multiselect()

setup = st.Page("pages/setup.py", title="0. Setup")
select_preferences = st.Page("pages/preferences.py", title="1. Select preferences")
handle_dtypes = st.Page("pages/handle_dtypes.py", title="2. Handle data types")
generator_selection = st.Page("pages/select.py", title="3. Generator selection")
missing_handling = st.Page("pages/missing_handling.py", title="4. Missing value handling")
generation = st.Page("pages/generate.py", title="5. Synthetic generation")
evaluation = st.Page("pages/evaluate.py", title="6. Evaluation")
rerun = st.Page("pages/rerun.py", title="Rerun workflow")
reports = st.Page("pages/report.py", title="Reports")

st.navigation(
    [
        setup,
        select_preferences,
        handle_dtypes,
        generator_selection,
        missing_handling,
        generation,
        evaluation,
        rerun,
        reports,
    ],
).run()

# TODO Add a homepage (explains the project and its features)
# TODO Add a workflow page (includes all steps besides setup which are currently present in navigation)

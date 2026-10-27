import shutil # Temporary while clear_output_ui exists
import time

import streamlit as st

from agentic_synth_data.utils import DATA_DIR, DEMO_DATA_DIR, PROJECT_ROOT
from agentic_synth_data.workflow import DataGen


st.sidebar.title("GuidedSDG")

st.write(
    "`● Setup ─── ○ Preference ─── ○ Data types ─── ○ Missing ───  ○ Generate ───  ○ Evaluate`"
)

st.header("About GuidedSDG")
st.write(
    "GuidedSDG is an agentic decision-support workflow for Synthetic Data "
    "Generation (SDG). A Large Language Model (LLM) agent assists users in "
    "preparing datasets, selecting suitable generators, and evaluating "
    "synthetic data based on the characteristics of the original dataset "
    "and user preferences."
)

st.write(
    "The application follows a structured workflow with human-in-the-loop "
    "interactions, allowing you to review and revise agent recommendations. "
    "Start by loading your dataset, which will be prepared for synthetic "
    "data generation."
)

drawer = st.popover("Click to view workflow outline", on_change="rerun")
with drawer:
    if drawer.open:
        with st.spinner("Loading outline..."):
            time.sleep(2) # Not sure whether to keep (otherwise blank when loading)
        st.write(
            "**Note:** Steps where `Agent` has a check mark `✓` indicate that the LLM agent is "
            "involved in providing suggestions to support generating your synthetic dataset."
        )
        st.write(
            "| Step | | Description | Agent |\n"
            "|------|---------|-------------|-----------|\n"
            "| 1 | | Load dataset and setup agent. | |\n"
            "| 2 | | Select preferences. | |\n"
            "| 3 | a) | Detect and standardize missing value markers. | |\n"
            "| | b) | Infer data types (Python and semantic). | ✓ |\n"
            "| | c) | Recommend generator (CTGAN, PATEGAN, BayesianNet, PrivBayes). | ✓ |\n"
            "| | d) | Apply missing value handling strategy (if needed). | ✓ |\n"
            "| 4 | a) | Save the prepared dataset. | |\n"
            "| | b) | Generate metadata for generator input. | |\n"
            "| 5 | | Run the selected synthetic data generator. | |\n"
            "| 6 | | Evaluate synthetic data against the original dataset. | ✓ |\n"
            "| 7 | | Generate a workflow report. | |\n"
        )

st.header("Load data and configuration")
st.space("small")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Load data")
    file = st.file_uploader("data", label_visibility="hidden",) # TODO: Increase max download size

    st.write("---")

    st.subheader("Try it out with a preset dataset: Titanic dataset")
    st.write("Try out GuidedSDG with a tester dataset to get an idea of how this tool works.")

    if "tester_selected" not in st.session_state or file is not None:
        st.session_state.tester_selected = False

    tester = st.button("Load Titanic dataset")

    if tester:
        st.session_state.tester_selected = True
        st.session_state.data_path = DEMO_DATA_DIR / "mix_titanic.csv"

        st.success("Titanic dataset loaded.")

with col2:
    st.subheader("Select engine")
    st.selectbox( # example
        "engine selection",
        ["MiniMaxAI/MiniMax-M2.5"],
        label_visibility="hidden",
    )
    st.subheader("Setup engine parameters")
    st.space("large")
    clicked = st.button("start", disabled=file is None and not st.session_state.tester_selected,)

if clicked:
    if st.session_state.tester_selected:
        data_path = st.session_state.data_path
        file_name = data_path.name

    else:
        data_dir = DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)

        data_path = data_dir / file.name

        with open(data_path, "wb") as f:
            f.write(file.getbuffer())

        file_name = file.name

    # Start a new workflow
    st.session_state.clear()

    gen = DataGen(data_path)
    gen.load_dataframe()

    st.session_state.gen = gen
    st.session_state.file_name = file_name

    st.switch_page("pages/preferences.py")


def clear_output_ui(): # Temporary fn
    """Delete all files and subdirectories inside output/."""
    output_dir = PROJECT_ROOT / "output"

    if st.button("Clear output folder", key="clear_output_button"):
        st.session_state.confirm_clear_output = True

    if st.session_state.get("confirm_clear_output", False):
        st.warning("This will permanently delete all files and folders inside output/.")

        if st.button("Confirm deletion", key="confirm_clear_output_button"):
            if output_dir.exists():
                for item in output_dir.iterdir():
                    if item.is_dir() and not item.is_symlink():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

            st.session_state.confirm_clear_output = False
            st.success("Output folder cleared.")

with st.sidebar:
    st.subheader("Developer tools")
    clear_output_ui()

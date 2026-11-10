import shutil # Temporary while clear_output_ui exists
import time

import streamlit as st

from agentic_synth_data.utils import DATA_DIR, DEMO_DATA_DIR, PROJECT_ROOT, \
    display_workflow_progress
from agentic_synth_data.workflow import DataGen


st.sidebar.title("GuidedSDG")

display_workflow_progress(
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
            time.sleep(2)

        st.write(
            "**Note:** Steps where `Agent` has a check mark `✓` indicate that the LLM agent is "
            "involved in providing suggestions to support generating your synthetic dataset.\n"
            "| Step | | Description | Agent |\n"
            "|------|---------|-------------|-----------|\n"
            "| 1 | | Load dataset and setup agent. | |\n"
            "| 2 | | Select preferences. | |\n"
            "| 3 | a) | Detect and standardize missing value markers. | |\n"
            "| | b) | Infer data types (Python and semantic). | ✓ |\n"
            "| | c) | Select covariates. | |\n"
            "| | d) | Train/test split. | |\n"
            "| | f) | Recommend generator (CTGAN, PATEGAN, BayesianNet, PrivBayes). | ✓ |\n"
            "| | g) | Apply missing value handling strategy (if needed). | ✓ |\n"
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
    st.subheader("Data source")

    data_source = st.radio(
        label="**Data source**",
        options=["Upload CSV", "Titanic demo"],
        horizontal=True,
    )

    file = None

    if data_source == "Upload CSV":
        file = st.file_uploader(
            "**Load a dataset**",
            type=["csv"],
            help="Maximum file size: 200 MB.",
        )
        st.caption(
            "Your data will only be used for the workflow and will not be saved anywhere."
        )

    else:
        st.write(
            "Try GuidedSDG with a sample Titanic dataset containing numerical and categorical "
            "features."
        )
        st.success("Titanic dataset selected.")

    if "tester_selected" not in st.session_state or file is not None:
        st.session_state.tester_selected = False

with col2:
    st.subheader("Language model")

    provider = st.selectbox(
        label="**Model provider**",
        options=["openai", "anthropic", "huggingface", "ollama", "lm-studio", "deepseek",],
        placeholder="Select a provider",
    )

    model_id = st.selectbox(
        label="**Model ID**",
        options=["MiniMaxAI/MiniMax-M2.5", "Enter manually",],
    )

    if model_id == "Enter manually":
        model_id = st.text_input(label=" ", placeholder="Enter the model identifier",)

    api_key = st.text_input(label="**API key**", placeholder="Enter the API key")

    st.caption(
        "**Note:** Your API key is used only to authenticate with the selected model provider. It "
        "is not stored permanently by GuidedSDG."
    )

    with st.expander("Advanced model settings"):
        base_url = st.text_input("API endpoint")
        # TODO: add supported model parameters

st.divider()

ready = (
    file is not None or data_source == "Titanic demo" and provider is not None and bool(model_id.strip())
)

if st.button("Start", disabled=not ready, type="primary",):
    if data_source == "Titanic demo":
        data_path = DEMO_DATA_DIR / "mix_titanic.csv"

    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data_path = DATA_DIR / file.name

        with open(data_path, "wb") as f:
            f.write(file.getbuffer())

    file_name = data_path.name

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

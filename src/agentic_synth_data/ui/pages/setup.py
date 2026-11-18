import time

import streamlit as st
from dotenv import dotenv_values

from agentic_synth_data.llm import check_model_connection
from agentic_synth_data.ui.helpers import clear_output_ui # NOTE: temporary
from agentic_synth_data.utils import DATA_DIR, DEMO_DATA_DIR, PROJECT_ROOT, comma_sep, \
    display_workflow_progress
from agentic_synth_data.workflow import DataGen


env_path = PROJECT_ROOT / ".env"
env = dotenv_values(env_path) if env_path.is_file() else {}


st.sidebar.title("GuidedSDG")

display_workflow_progress(
    "`● Setup ─── ○ Preference ─── ○ Data types ─── ○ Missing ───  ○ Generate ───  ○ Evaluate`"
)

st.header("About GuidedSDG")
st.write(
    "GuidedSDG is an agentic decision-support workflow for Synthetic Data Generation (SDG). A "
    "Large Language Model (LLM) agent assists users in preparing datasets, selecting suitable "
    "generators, and evaluating synthetic data based on the characteristics of the original "
    "dataset and user preferences."
)

st.write(
    "The application follows a structured workflow with human-in-the-loop interactions, allowing "
    "you to review and revise agent recommendations. Start by loading your dataset, which will be "
    "prepared for synthetic data generation."
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
            "| 1 | | Load dataset and configure LLM agent. | |\n"
            "| 2 | | Select synthesis preferences. | |\n"
            "| 3 | a) | Detect and standardize missing value markers. | |\n"
            "| | b) | Infer Python data types and semantic types. | ✓ |\n"
            "| | c) | Select covariates. | |\n"
            "| | d) | Split data into training and test sets. | |\n"
            "| | e) | Recommend generator (CTGAN, PATEGAN, BayesianNet, PrivBayes). | ✓ |\n"
            "| | f) | Recommend and apply a missing value handling strategy (if necessary). | ✓ |\n"
            "| 4 | a) | Save the prepared dataset. | |\n"
            "| | b) | Generate metadata for generator input. | |\n"
            "| 5 | | Generate synthetic data using the selected generator. | |\n"
            "| 6 | a) | Calculate fidelity and privacy evaluation metrics. | |\n"
            "| | b) | Interpret evaluation results and recommend improvements. | ✓ |\n"
            "| | c) | Generate a workflow report. | |\n"
            "| 7 | | Optionally revise workflow decisions (repeat generation and evaluation). | ✓ |"
        )

st.header("Load data and configuration")
st.space("small")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Language model")

    providers = ["openai", "ollama", "lm-studio"]
    env_provider = env.get("MODEL_PROVIDER")

    if env_provider and env_provider not in providers:
        st.warning(
            f"The provider '{env_provider}' configured in .env is not supported by the Streamlit "
            "interface. Please select a supported provider.\nSupported providers: "
            f"{comma_sep(providers)}"
        )

    provider = st.selectbox(
        label="**Model provider**",
        options=providers,
        index=providers.index(env_provider)
        if env_provider in providers else 0,
    )

    env_model = env.get("MODEL_ID")

    model_options = (
        [env_model, "Enter manually"]
        if env_model and provider == env_provider
        else ["Enter manually"]
    )

    model_id = st.selectbox(label="**Model ID**", options=model_options,)

    if model_id == "Enter manually":
        model_id = st.text_input(
            label="**⇨ Manual model identifier entry**",
            placeholder="Enter the model identifier",
        )

    api_key = None

    if provider == "openai":
        env_key = env.get("OPENAI_API_KEY") if provider == env_provider else None

        api_key_input = st.text_input(
            label="**API key**",
            type="password",
            placeholder="Configured in .env" if env_key else "Enter API key",
        )

        api_key = api_key_input or env_key

        st.caption(
            "**Note:** Your API key is used only to authenticate with the selected model provider. "
            "It is not stored permanently by GuidedSDG."
        )

    with st.expander("Advanced model settings"):
        env_url = env.get("BASE_URL") if provider == env_provider else None

        base_url = st.text_input(
            label="**API endpoint**", value=env_url or "", placeholder="Optional",
        ) or None

with col2:
    st.subheader("Data source")

    data_source = st.radio(
        label="**Data source**",
        options=["Upload CSV", "Titanic demo"],
        horizontal=True,
        #label_visibility="hidden",
    )

    file = None

    if data_source == "Upload CSV":
        file = st.file_uploader(
            "**Load a dataset**",
            type=["csv"],
            help="Maximum file size: 200 MB.",
        )

        st.caption(
            "**Note:** Your data will only be used for the workflow and will not be saved anywhere."
        )

    else:
        st.write(
            "Try **GuidedSDG** with a sample Titanic dataset containing numerical and categorical "
            "features."
        )

        st.success("Titanic dataset selected.")

    if "tester_selected" not in st.session_state or file is not None:
        st.session_state.tester_selected = False

    ready = (
        (file is not None or data_source == "Titanic demo") and
        bool(model_id.strip())
    )

    _, _, col3 = st.columns(3)

    with col3:
        st.space("medium")

        start = st.button("Start", disabled=not ready, type="primary",)

if start:
    if data_source == "Titanic demo":
        data_path = DEMO_DATA_DIR / "mix_titanic.csv"

    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data_path = DATA_DIR / file.name

        with open(data_path, "wb") as f:
            f.write(file.getbuffer())

    file_name = data_path.name

    with st.spinner("Checking model connection..."):
        gen = DataGen(
            data_path,
            model_id=model_id,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )

        connected, message = check_model_connection(gen.model)

    if not connected:
        st.error(message)
        st.stop()

    try:
        gen.load_dataframe()

    except Exception as exc:
        st.error(f"Dataset loading failed: {exc}")
        st.stop()

    # Start a new workflow
    st.session_state.clear()

    st.session_state.gen = gen
    st.session_state.file_name = file_name

    st.switch_page("pages/preferences.py")

with st.sidebar:
    st.subheader("Developer tools")
    clear_output_ui()

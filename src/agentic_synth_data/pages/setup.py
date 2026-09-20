from pathlib import Path
import time

import streamlit as st

from agentic_synth_data.workflow import DataGen


st.sidebar.title("GuidedSDG")

st.write(
    "`● Setup ─── ○ Preference ─── ○ Data types ─── ○ Missing ───  ○ Generate ───  ○ Evaluate`"
)

st.header("About GuidedSDG")
st.write(
    "GuidedSDG is an agentic workflow for Synthetic Data Generation (SDG) "
    "where a Large Language Model (LLM) agent will assist you with generating "
    "usable synthetic data, based on the characteristics of your original dataset. "
)
st.write(
    "This tool follows a set workflow where you can load your dataset, "
    "which will be prepared for synthetic data generation."
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
            "| 7 | | Generate a workflow report. | ✓ |\n"
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

    tester = st.button("Load Titanic dataset")

    if tester:
        st.session_state.tester_selected = True
        data_path = "../data/mix_titanic.csv"

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
    data_dir = Path("../data")
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / file.name

    with open(data_path, "wb") as f:
        f.write(file.getbuffer())

    # Reset state from previous workflow
    # TODO Update to clear all (now versions are added, maybe just list all)
    for key in [
        "semantic_types_complete",
        "semantic_types",
        "semantic_types_inference",
        "semantic_type_confirmation",
        "generator_feedback_hist",
        "generator_plan",
        "pending_generator_review",
        "generator_selection_complete",
        "generation_complete",
        "evaluation_review_history",
        "attack_metrics"
        "agent_evaluation",
        "generator_revision_plan",
        "missing_revision_plan",
    ]:
        st.session_state.pop(key, None)

    gen = DataGen(data_path)
    gen.load_dataframe()

    st.session_state.gen = gen
    st.session_state.file_name = file.name

    st.switch_page("pages/preferences.py")

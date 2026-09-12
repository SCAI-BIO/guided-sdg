from pathlib import Path
import streamlit as st

from agentic_synth_data.workflow import DataGen


st.sidebar.title("Agentic synthetic data workflow")

st.write("● Setup ─── ○ Preference ─── ○ Data types ─── ○ Missing ───  ○ Generate ───  ○ Evaluate")
st.write("---")
st.header("Load data and configuration")
st.space("small")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Load data")
    file = st.file_uploader("data", label_visibility="hidden",) # TODO: Increase max download size

with col2:
    st.subheader("Select engine")
    st.selectbox( # example
        "engine selection",
        ["MiniMaxAI/MiniMax-M2.5"],
        label_visibility="hidden",
    )
    st.subheader("Setup engine parameters")
    st.space("large")
    clicked = st.button("start", disabled=file is None,)

if clicked:
    data_dir = Path("../data")
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / file.name

    with open(data_path, "wb") as f:
        f.write(file.getbuffer())

    # Reset state from previous workflow
    for key in [
        "semantic_types_complete",
        "semantic_types",
        "semantic_types_inference",
        "semantic_type_confirmation",
    ]:
        st.session_state.pop(key, None)

    gen = DataGen(data_path)
    gen.load_dataframe()

    st.session_state.gen = gen
    st.session_state.file_name = file.name

    st.switch_page("./pages/preferences.py")

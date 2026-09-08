import tempfile
import streamlit as st

from agentic_synth_data.workflow import DataGen

st.write("✓ Setup ─── ● Preference ─── ○ Data types ─── ○ Missing ───  ○ Generate ───  ○ Evaluate")
st.write("---")
st.header("Select preferences")
st.space("small")

st.sidebar.title("Agentic synthetic data workflow")

# Configuration
st.header("Load data and configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Load data")
    file = st.file_uploader("Data") # TODO: Increase max download size

with col2:
    st.subheader("Select engine")
    # dropdown menu?
    #st.menu_button("Message", options=[])
    st.subheader("Setup engine parameters")
    # dropdown menu?
# Additional session information (possibly?)
    st.space("large")
    clicked = st.button("start", disabled=file is None,)

if clicked:
    with tempfile.NamedTemporaryFile(
        delete=False,
    ) as tmp:
        tmp.write(file.getbuffer())
        data_path = tmp.name

    gen = DataGen(data_path)

    gen.load_dataframe()

    st.session_state.gen = gen
    st.session_state.file_name = file.name

    st.switch_page("./pages/select_preferences.py")

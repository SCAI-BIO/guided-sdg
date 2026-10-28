from pathlib import Path

import streamlit as st

from agentic_synth_data.rerun_workflow import version_tabs_ui
from agentic_synth_data.utils import OUTPUT_DIR


st.sidebar.title("GuidedSDG")

st.write(
    "`✓ Preferences ─ ✓ Data types ─ ✓ Generator ─ ✓ Missing values ─ ✓ Generate ─ ✓ Evaluate`"
)

st.header("Workflow reports")
st.write("Review SDG workflow decisions and results.")
st.space("small")

# TODO: Include report comparison as well.

revision_tabs = version_tabs_ui(label="Version", key="active_report_tab", include_initial=True,)

for version_id, tab in revision_tabs.items():
    if not tab.open:
        continue

    with tab:
        revision = st.session_state.revision_history.get(version_id)

        if revision is None:
            st.warning(f"Version {version_id} was not found.")
            continue

        data_name = Path(revision.data_path).stem

        report_path = OUTPUT_DIR / f"version_{version_id}" / f"report_{data_name}_v{version_id}.md"

        if report_path.exists():
            report_content = report_path.read_text(encoding="utf-8")
            st.markdown(report_content)
            st.download_button(
                label="Download report",
                data=report_content,
                file_name=report_path.name,
                mime="text/markdown",
                key=f"download_report_v{version_id}",
            )

        else:
            st.info(f"No report available for version {version_id}.")

# TODO: Add a method to compare reports...
    # For each report, add a checkbox to select which version reports to be compared
        # st.checkbox("Are these correct?")
        # Maybe set a max of 3? (Perhaps start without limiting it for now though...)

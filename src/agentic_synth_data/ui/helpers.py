import shutil # NOTE: Temporary while clear_output_ui exists

import streamlit as st

from agentic_synth_data.utils import PROJECT_ROOT


class CapturedOutput:
    """Capture text written to a redirected output stream."""
    def __init__(self):
        self.buffer = ""


    def write(self, text: str | bytes) -> None:
        """Append text to the captured output buffer."""
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        elif not isinstance(text, str):
            text = str(text)

        self.buffer += text

    def flush(self) -> None:
        pass


class StreamlitOutput(CapturedOutput):
    """Capture text output and optionally display it live in Streamlit."""
    def __init__(self, live: bool = True):
        super().__init__()
        self.placeholder = st.empty() if live else None


    def write(self, text: str | bytes) -> None:
        """Capture text and optionally display the output."""
        super().write(text)

        if self.placeholder is not None:
            self.placeholder.code(self.buffer)


def clear_output_ui(): # NOTE: Temporary fn
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

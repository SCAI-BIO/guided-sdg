import streamlit as st


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

import streamlit as st


class StreamlitOutput:
    def __init__(self, live: bool = True):
        self.live = live
        self.placeholder = st.empty()
        self.buffer = ""

    def write(self, text):
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        elif not isinstance(text, str):
            text = str(text)

        self.buffer += text

        if self.live and self.placeholder is not None:
            self.placeholder.code(self.buffer)

    def flush(self):
        pass

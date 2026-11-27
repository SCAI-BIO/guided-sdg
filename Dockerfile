FROM python:3.13

LABEL org.opencontainers.image.source=https://github.com/SCAI-BIO/guided-sdg
LABEL org.opencontainers.image.licenses=GPL-3.0-or-later
WORKDIR /usr/local/app

# Install the application dependencies
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock LICENSE README.md ./

# Copy in the source code
COPY src ./src
COPY demo_data ./demo_data

RUN uv sync --frozen --no-dev

EXPOSE 8080

CMD ["uv", "run", "streamlit", "run", "src/agentic_synth_data/ui/main.py", "--server.address=0.0.0.0", "--server.port=8080"]

FROM python:3.13
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

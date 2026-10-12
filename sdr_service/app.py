"""Thin HTTP wrapper around the spring-epfl/synthetic_data_release CLI.

The official image is built to be driven from the command line. Running that
CLI from inside the guided-sdg container would require podman/docker access,
which is unavailable (and undesirable) inside a container. Instead this
service runs as an always-on sidecar next to guided-sdg and executes the
generator inside its own container, where the dependencies already live.

Both containers share the same data/ and output/ volumes, so the paths sent
here are the same paths the app uses (no translation required).
"""

from pathlib import Path
import subprocess
import os
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="synthetic-data-release service")

SDR_WORKDIR = Path(os.environ.get("SDR_WORKDIR", "/synthetic_data_release"))
GENERATOR_SCRIPT = "executables/generate_synthetic_dataset.py"


class GenerateRequest(BaseModel):
    mechanism: str
    datapath: str
    output_file: str
    sample_size: int
    parameters: Optional[str] = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
def generate(request: GenerateRequest) -> JSONResponse:
    command = [
        "python3",
        GENERATOR_SCRIPT,
        "-D", request.datapath,
        "-M", request.mechanism,
    ]

    if request.parameters:
        command += ["-P", request.parameters]

    command += [
        "-O", request.output_file,
        "-N", str(request.sample_size),
    ]

    process = subprocess.run(
        command,
        cwd=SDR_WORKDIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    body = {
        "returncode": process.returncode,
        "log": process.stdout,
        "output_file": request.output_file,
    }

    if process.returncode != 0:
        return JSONResponse(status_code=500, content=body)

    return JSONResponse(status_code=200, content=body)

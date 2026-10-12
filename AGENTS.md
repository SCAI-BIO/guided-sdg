# AGENTS.md

Agentic workflow for synthetic data generation and evaluation. Streamlit UI plus an interactive CLI, driven by a LangChain/LangGraph LLM agent. Python 3.13, `uv`-managed, src-layout package `agentic_synth_data`.

## Commands

```bash
uv sync                                              # install deps
uv run streamlit run src/agentic_synth_data/main.py  # run the UI (repo root)
```

- There is **no test suite, linter, formatter, typecheck, or CI** configured. Do not invent one; if you add tooling, wire it up explicitly.
- The CLI path (`DataGen.run()`) uses `pyinputplus` and is **interactive only** — it cannot be driven non-interactively. Use the notebook `notebooks/demo-workflow.ipynb` to exercise it step by step.

## Environment

- `.env` at repo root (loaded in `main.py`) needs `OPENAI_API_KEY` and `BASE_URL` (OpenAI-compatible endpoint).
- The model is **hardcoded** to `MiniMaxAI/MiniMax-M2.5` in two places: `workflow.py` (`DataGen.create_model`) and `pages/setup.py`. Change both if switching models.

## Architecture

- `main.py` registers the Streamlit pages in `src/agentic_synth_data/pages/` via `st.navigation`.
- `workflow.py` defines `DataGen`, the workflow engine. It is the hub: every step has a CLI method and a `_ui` twin (e.g. `preprocess_dtypes` / `preprocess_dtypes_ui`). **When changing a step, update both paths** — they duplicate logic, differing mainly in `print` vs Streamlit output.
- UI pages share state through `st.session_state.gen` (a single `DataGen` instance), created in `pages/setup.py`. Pages call the corresponding `_ui` methods.
- Step order: load → preferences → dtype/semantic handling → generator selection → missing-value handling → save prepared + metadata → synthesize → evaluate → revise.
- Subpackages: `dataset_preparation/` (dtypes, semantic types, missing values), `synthetic_generation/` (generator profile/selection), `strategies/` (generator backends), `dataset_evaluation/` (syndat + anonymeter metrics), `prompts/`, `schemas/`, `pages/`.
- `utils.PROJECT_ROOT` is the repo root (parents[2] of `utils.py`); prefer it over relative paths.

## Generator execution (non-obvious)

- Generators are **not** run in-process. `strategies/synthetic_data_release.py` POSTs to the always-on `sdr` sidecar (`SDR_URL`, default `http://localhost:8000`), a thin FastAPI wrapper (`sdr_service/app.py`, `Dockerfile.sdr`) around the `springepfl/synthetic-data` image that runs `executables/generate_synthetic_dataset.py` inside its own container. `compose.yaml` starts both services.
- All four generators (CTGAN, PATEGAN, BayesianNet, PrivBayes) use the `synthetic_data_release` backend (`strategies/factory.py`).
- Output paths: `utils.DATA_DIR` (prepared CSV + co-located metadata JSON) and `utils.OUTPUT_DIR` (`version_<N>/` synthetic CSV and plots/report). Both default to `PROJECT_ROOT/{data,output}` and are overridable via `DATA_DIR`/`OUTPUT_DIR` env vars (set to `/workspace/...` in compose so the app and `sdr` agree on paths).

## Known issues (verify before relying on these areas)

- `main.py` registers `pages/report.py`, which does not exist (only the package-root `report.py`); the Reports nav entry is broken.
- `notebooks/demo-workflow.ipynb` calls `gen.select_preferences()`, but that method no longer exists — it is now `DataGen.apply_preferences()`.


# GuidedSDG

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="src/agentic_synth_data/images/icon-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="src/agentic_synth_data/images/icon.png">
  <img alt="guided-sdg-image" src="src/agentic_synth_data/images/icon.png">
</picture>

GuidedSDG is a Large Language Model (LLM) assisted workflow for synthetic tabular data generation and evaluation.

The application guides users through data preparation, generator selection, synthetic data generation, and evaluation. It incorporates human-in-the-loop (HITL) interactions, allowing users to review and revise agent recommendations throughout the workflow.


## Getting started


### 1. Prerequisites


#### 1.1 Install the project dependencies

Verify that [UV](https://docs.astral.sh/uv/getting-started/installation/) is installed. 
Enter the following into the terminal:
```bash
uv sync
```


#### 1.2 Install one of the following container engines

Verify that one of the following is on your system:

- [Podman](https://podman.io/) with [Podman Compose](https://github.com/containers/podman-compose#installation)

- [Docker](https://docs.docker.com/get-docker/) with [Docker Compose](https://docs.docker.com/compose/install/)

Verify the installation:

```bash
# If Podman and Podman-Compose is installed:
podman --version
podman compose version

# If Docker and Docker-Compose is installed:
docker --version
docker compose version
```


#### 2. Configure environment variables
Create a `.env` file with the necessary API credentials for the configured language model.
- Note: OpenAI compatible APIs are currently permitted.

```python
# Supported Streamlit providers: openai, ollama, lm-studio
MODEL_PROVIDER=openai
MODEL_ID=your-model-identifier

# Authentication (required for hosted OpenAI-compatible APIs)
OPENAI_API_KEY=your-api-key

# API endpoint (optional if using the provider's default)
BASE_URL=https://your-provider-endpoint/v1
```


#### 3. Start the application

The application uses two containers:

**guided-sdg**: Runs the Streamlit interface and workflow.

**sdr**: Runs the Synthetic Data Release generators through an HTTP API.

```bash
# Using Podman:
podman build -t guided-sdg .
podman compose up -d --build

# Using Docker:
docker build -t guided-sdg .
docker compose up -d --build
```

Click [here](http://localhost:8080) to access the Streamlit interface after running the container.

To stop the containers:
```bash
# Using Podman:
podman compose down

# Using Docker:
docker compose down
```


#### 4. Start the Streamlit user interface
Enter the following into the terminal:
```bash
streamlit run ./src/agentic_synth_data/ui/main.py
```

```bash
podman run --rm --env-file .env -p 8080:8080 guided-sdg
```

``` bash
# If podman is installed:
podman-compose up # To start the containers
podman-compose down # To stop and remove services

# If docker is installed:
docker compose up # ""
docker compose down # ""
```
---

## Workflow:

| Step | | Description | LLM assistance |
|------|---------|-------------|-----------|
| 1 | | Load dataset and configure LLM agent. | |
| 2 | | Select synthesis preferences. | |
| 3 | a) | Detect and standardize missing value markers. | |
| | b) | Infer Python data types and semantic types. | ✓ |
| | c) | Select covariates. | |
| | d) | Split data into training and test sets. | |
| | e) | Recommend generator (CTGAN, PATEGAN, BayesianNet, PrivBayes). | ✓ |
| | f) | Recommend and apply a missing value handling strategy (if necessary). | ✓ |
| 4 | a) | Save the prepared dataset. | |
| | b) | Generate metadata for generator input. | |
| 5 | | Generate synthetic data using the selected generator. | |
| 6 | a) | Calculate fidelity and privacy evaluation metrics. | |
| | b) | Interpret evaluation results and recommend improvements. | ✓ |
| | c) | Generate a workflow report. | |
| 7 | | Optionally revise workflow decisions and repeat generation and evaluation. | ✓ |

*Info:* Each check mark (✓) indicates that the LLM agent is invoked to assist during the workflow. The agent is only involved in making suggestions.

**Note:**
- Using generative models from [Synthetic Data Release](https://github.com/spring-epfl/synthetic_data_release).
- Using synthetic data evaluation metrics from [syndat](https://github.com/SCAI-BIO/syndat)
- Using privacy risk evaluation metrics from [anonymeter](https://github.com/statice/anonymeter)
- <a href="https://www.flaticon.com/free-icons/chart" title="chart icons">Chart icons created by mynamepong - Flaticon</a>

## Project structure
```bash
guided-sdg
├── compose.yaml # Container configuration
├── .env # LLM API configuration
├── data # Input and prepared datasets
├── demo_data # Example dataset
│   └── mix_titanic.csv
├── notebooks # Example workflow notebook
├── output # Versioned generation results
│   └── version_1
│       ├── synthetic_*_*_v*.csv # Generated synthetic dataset
│       ├── plots # Evaluation visualizations
│       └── report_*.md # Workflow report
├── sdr_service # Synthetic data generator service
└── src
    └── agentic_synth_data
```

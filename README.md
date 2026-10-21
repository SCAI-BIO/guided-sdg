
# GuidedSDG

**An agentic decision-support workflow for Synthetic Data Generation (SDG)**


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
OPENAI_API_KEY="your-key-here"
BASE_URL="https://your-openai-compatible-endpoint/v1"
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

Access the Streamlit interface at:

http://localhost:8080

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
streamlit run ./src/agentic_synth_data/main.py
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
| 1 | | Load dataset. | |
| 2 | | Select preferences. | |
| 3 | a) | Detect and standardize missing value markers. | |
| | b) | Infer data types (Python and semantic). | ✓ |
| | c) | Recommend generator (CTGAN, PATEGAN, BayesianNet, PrivBayes). | ✓ |
| | d) | Apply missing value handling strategy (if needed). | ✓ |
| 4 | a) | Save the prepared dataset. | |
| | b) | Generate metadata JSON for generator input. | |
| 5 | | Run the selected synthetic data generator. | |
| 6 | | Evaluate synthetic data against the original dataset. | ✓ |
| 7 | | Generate a workflow report. | |

*Info:* Each check mark (✓) indicates that the LLM agent is invoked to assist during the workflow. The agent is only involved in making suggestions.

**Note:**
- Using generative models from [Synthetic Data Release](https://github.com/spring-epfl/synthetic_data_release).
- Using synthetic data evaluation metrics from [syndat](https://github.com/SCAI-BIO/syndat)
- Using privacy risk evaluation metrics from [anonymeter](https://github.com/statice/anonymeter)


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

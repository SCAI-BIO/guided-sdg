
# agentic-synth-data
Agentic workflow for synthetic data generation &amp; evaluation.

## Instructions to access the user interface:
### 1. Install the project dependencies
Verify that [UV](https://docs.astral.sh/uv/getting-started/installation/) is installed. 
Enter the following into the terminal:
```bash
uv sync
```

### 2. Configure environment variables
Create the needed `.env` file with the necessary API credentials for the configured language model.
- Note: OpenAI compatible APIs are currently permitted.

```bash
OPENAI_API_KEY="your-key-here"
BASE_URL="https://your-openai-compatible-endpoint/v1"

```

### 3. Start the Streamlit user interface
Enter the following into the terminal:
```bash
streamlit run ./src/agentic_synth_data/main.py
```

---

## Workflow:

| Step | | Description | Agent |
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
| 7 | | Generate a workflow report. | ✓ |

Note:
- Using generative models from [Synthetic Data Release](https://github.com/spring-epfl/synthetic_data_release).
- Using synthetic data evaluation metrics from [syndat](https://github.com/SCAI-BIO/syndat)
- Using privacy risk evaluation metrics from [anonymeter](https://github.com/statice/anonymeter)
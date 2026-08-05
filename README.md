
# agentic-synth-data
Agentic workflow for synthetic data generation &amp; evaluation.

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
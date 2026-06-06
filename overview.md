# Agentic synthetic data 

## Workflow:

| Step | | Description | Agent |
|------|---------|-------------|-----------|
| 1 | | Load dataset. | |
| 2 | | Select preferences. | |
| 3 | a) | Detect and standardize missing value markers. | |
| | b) | Infer data types (Python and semantic). | ✓ |
| | c) | Recommend generator (CTGAN, PATEGAN, BayesianNet, PrivBayes). | ✓ |
| | d) | Recommend and apply missing value handling strategy. | ✓ |
| 4 | a) | Save the prepared dataset. | |
| | b) | Generate metadata JSON for generator input. | |
| 5 | | Run the selected synthetic data generator. | |
| 6 | | Evaluate synthetic data against the original dataset. | ✓ |
| 7 | | Generate a workflow report. | ✓ |
---
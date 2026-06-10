
# agentic-synth-data
Agentic workflow for synthetic data generation &amp; evaluation.

## Meeting notes:
### From last time: Generators
* Can handle missing values:
  * BayesianNet
  * PrivBayes
* Cannot handle missing values:
  * PATEGAN
  * CTGAN
*Still inconsistent (sometimes tools don't work, possibly due to (semantic) data type encoding).

---
## Previous meetings:
### First:
- [ ] Include agent in crashes in the workflow to try to fix it.
  - [ ] Should add a check that all required information is included if impute is selected for missing value handling, etc.
  - NOTE: Added for JSON output (have to test, maybe in the instructions tell it to do something wrong and see if it fixes it (but contradictory))
  - [x] Feed validation errors to the agent (JSON output) to fix. -> Did not work with my quick test.
    - `Test`: in stype_prompt, stated to include "```json", hoping the error and formatting message would be sufficient for the agent to correct itself, but no (but llms tend to struggle with contradictory instruction, have to try finding this case "in the wild" to verify)
    - `NOTE`: Currently only runs once (then raises).
  - Common steps: missing value handling (num and cat handling), generation (possibly issue with semantic types selected), evaluation (dtype mismatch, I think mostly resolved removing categorical columns from metrics).
    - `NOTE`: Categorical columns are currently not evaluated thoroughly (cat_col_drop etc.).
- [x] `generator_prompt`: Add correlations and summary statistics (to inform selection).
  - `NOTE`: Categorical columns aren't covered.
  - `NOTE`: Added nunique (int) for cols as a quick cardinality check.
  - `NOTE`: Included skew in summary statistics, and added a threshold (absolute max. skew) to determine whether Pearson or Spearman's correlation should be used.
- [x] `evaluation_prompt`: Add information for the new SYNDAT metrics included (Normalized correlation difference and Discriminator AUC).
  - `NOTE`: Also, added information on steps, so agent has more clear recommendation strategy.
  - `NOTE`: Want to add SYNDAT print statement outputs.
  - [x] Give more context on the workflow steps.
    - Ex: The data has followed a deterministic workflow until this point (description of the workflow). Now, only if necessary, you are able to retrace the following previous steps in the workflow: missing value handling (change missing value handling strategy; not yet implemented) and generate synthetic data (want to implement first).
    - Possibly could be limited to something like "Currently, the only change you can call is returning to the generation step and choosing a new generator. Do this if you see it is needed. The user will approve if the decision is warranted".  
    - `NOTE`: Now need to have a greater focus on how to implement this (i.e., updating the synthesis request, documenting which changes were made (and why), and how actual steps should be revisited).

### Second:
- [x] Include a message indicating agent will respond (e.g., if discuss or another choice is selected, show the workflow is still running).
  - Example: "Agent is considering information based on your update"
- [x] Look into hyperparameter tuning.
  - Most likely will have to implement Optuna
  - [x] Check if [Synthetic Data Release](https://github.com/spring-epfl/synthetic_data_release) includes this intrinsically (circumvents using Optuna).
    - `NOTE` for evaluation step: Has attack models (to evaluate raw patient data leakage), might be helpful, in addition to SYNDAT (look into).
      - Has a membership inference attack and attribute inferences attack (inferring a target's sensitive value given information about the target record).
  - Inclusion of summary statistics could help (model can likely derive how to set up hyperparameters based on this) [in select_generator]: Haven't yet included hyperparam setup.
- [ ] Include Real4Reg tools: HIVAE could be directly included.
- Small publication/speech/poster etc.


### More on revisiting workflow steps:
Tool for the agent to call with the following features:
- `Possibly at the end of the eval`: Confirm agent's assessment (successful, needs review, not_successful).
  - If successful and user agrees: stop
  - Otherwise (F):
    - `retry_steps`: Two main options to redirect to previous steps
      1. Use existing methods and re enter them (user has to manually approve everything again and possibly set up, depending on what the agent remembers).
      2. The agent already processes and retries previously completed steps with new plan (after user has approved of agent suggestion, but could be more hands-off than desired for the user).
        - Would have to make tools for the agent to run the necessary steps again in a way that doesn't impede with the workflow.
  - `NOTE`: The report should be updated, stating which step was rerun and update the final report with the appended additions to the workflow. 
  - Allow for agent to compare both stages and agent suggests which version to keep/use. 
* Main idea: make specific steps into functions that the agent can call.

Mentioned before:
  - Subsequent runs, regenerate the objects (datagen1, datagen2, datagen3,... etc.) and store the results nad evaluated based on this (doesn't have to be an array).
---

`missing_value_handling`: agent consistently disregarded including a way to handle numerical values (added more explicitly to the prompt that both cat and num must be added, but a more explicit check is warranted).
  - [ ] Implement a way to detect certain values.

### TODO (moved, but mentioned last time):
- [ ] Prompted explanation of metrics `evaluation`.
- [ ] Preprocessing `TODO`: Add messages indicating the the df was updated (save previous version).
  - `NOTE`: Might be best when/if History is introduced.
- `generator_prompt` (correlations and summary statistics):
  - [ ] Add a disclaimer or if local or cloud-based model, state "will not be privacy exposing and print a warning"
    - [ ] Either detect or have the user indicate global model use in setup.


- Imputation options are very limited (sklearn leans toward numerical)
--- 

### Note on data generator parameters:
in `generate_synthetic_dataset.py`:
* BayesianNet parameters:
  * histogram_bins=10, 
  * degree=1, 
  * infer_ranges=False, 
  * multiprocess=True, 
  * seed=None
* PrivBayes parameters:
  * histogram_bins=10, 
  * degree=1, 
  * epsilon=.1, (Only difference b/t BayesianNet)
  * infer_ranges=False, 
  * multiprocess=True, 
  * seed=None
* CTGAN parameters:
  * embedding_dim=128, 
  * gen_dim=(256, 256), 
  * dis_dim=(256, 256), 
  * l2scale=1e-6, 
  * batch_size=500, 
  * epochs=300, 
  * multiprocess=False
* PATEGAN parameters:
  * eps=1, 
  * delta=1e-5, 
  * infer_ranges=False, 
  * num_teachers=10, 
  * n_iters=100, 
  * batch_size=128, 
  * learning_rate=1e-4, 
  * multiprocess=False

#### Example parameter settings:
```bash
-P "128:(256,256):(256,256):1e-6:500:300:False"
```

---

## Workflow:
1. Load dataframe (verify valid format for pandas)

2. Select preferences

3. Preprocess data

    a) Check for missing value markers

    b) Handle dtypes (two categories):
      - Python data types: int, str, float, etc.
      - Semantic data types: categorical, ordinal, integer, boolean, etc.

    c) Select/verify chosen generator:
      - PATEGAN
      - CTGAN
      - BayesianNet
      - PrivBayes

    d) Handle missing values (if needed for the generator)

4. Save information for synthetic generator tool use:

    a) Save prepared dataset

    b) Create metadata JSON for tool use

5. Run the selected data generator

6. Evaluate metrics comparing the synthetic data to the original data (or missing value marker replaced data)
7. Generate report

<br>

Using generative models from [Synthetic Data Release](https://github.com/spring-epfl/synthetic_data_release).
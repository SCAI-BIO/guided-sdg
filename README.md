
# agentic-synth-data
Agentic workflow for synthetic data generation &amp; evaluation.

## Meeting notes:
### Generators
* Can handle missing values:
  * BayesianNet
  * PrivBayes
* Cannot handle missing values:
  * PATEGAN
  * CTGAN

### Changes:
* Generator mechanism selection is now done separately from running the tool (no agent tool now, possibly implement for redesigning the workflow). 
  * select_generator
  * run_generator
* Handling missing values is now optional based on the data generator selected.
* Added additional metrics to the evaluation (Normalized correlation difference and Discriminator AUC).
  * `NOTE`: Requires additional information.
  
### Last meeting:
- [x] Move generator selection to before handling missing values. If missing values are needed to be removed, missing_value_handling processes it (processed_df), otherwise the handle_dtypes_df is returned as the processed_df.
  - `NOTE`: BayesianNet and PrivBayes were the only models that allow for missing values.
    - PATEGAN: `ValueError: NaN not in list.`
- [x] Output a report with specifications of what treatments were done with the data as a txt or JSON.
- [x] Make edit more clear as to what it means in generator selection.
- [x] handle_semantic_types: give the agent the number of unique values alongside the unique value (e.g., a list of tuples [(1, 2), ("string", 39), ...], or a dictionary {1: 2, "string"}) (a little bit than just the number of unique values).
- [x] Try defining a Pydantic schema for validation of structured outputs (see if this works).
  - [ ] Cool to implement, have the agent give the value error until it is right.
    - `NOTE`: Had value errors in the evaluation step with certain models.
- [ ] Check, mess up the dtype of a column and see what the agent detects.
    - `NOTE`: Altering the dtype (python) results in an error in the tool output.

### Not yet implemented:
- [ ] Revisit previous workflow steps.
- [ ] Evaluation prompt: give more context on the workflow (limit to something like "Currently, the only change you can call is returning to the generation step and choosing a new generator. Do this if you see it is needed. The user will approve if the decision is warranted".)  
   - Ex: The data has followed a deterministic workflow until this point (description of the workflow). Now, only if necessary, you are able to retrace the following previous steps in the workflow: missing value handling (change missing value handling strategy; not yet implemented) and generate synthetic data (want to implement first).
   - [ ] Prompted explanation of metrics.
- [ ] Preprocessing `TODO`: Add messages indicating the the df was updated (save previous version).
  - `NOTE`: Might be best when/if History is introduced.

### More on revisiting workflow steps:
Tool for the agent to call with the following features:
- Should allow for user confirmation of the agent's assessments.
- Should allow for the display and editing of steps
- Should finally redirect to previous steps (not sure if confirmations with user are necessary or not when rerun)
  - `NOTE`: the report that is generated should be updated stating which step was rerun and update the final report with the appended additions to the workflow. 
  - Possibly then allow for the agent to make a comparison between both stages and agent suggests which version to keep (this would likely be good to update the final report with).
  
* Main idea: make specific steps into functions that the agent can call.

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

## Data synthesis workflow (plan):

1. Load dataframe (verify valid format for pandas)

2. Select preferences

3. Preprocess data

    a) Check for missing value markers

    b) Handle dtypes (both categories)

    c) Select/verify chosen generator

    d) Handle missing values (if needed for the generator)

4. Save information for synthetic generator tool use:

    a) Save prepared dataset

    b) Create metadata (JSON) for tool use

5. Run the selected data generator

6. Evaluate metrics comparing the synthetic data to the original data (or missing value marker replaced data)

---

### TODOs
1. Load dataframe (verify valid format for pandas)
   - `TODO`: Add a dataset size minimum (the tools need about 150 samples, if I am remembering right).
     - Check min number of samples needed to run (would also need to disallow removal or thresholding (update to lower threshold or just impute))

2. Select preferences
   - possible `TODO`: Setup the dictionary in the nested format (not consistent in preference selection). 
   - possible `TODO`: Clarify possible confusion if mode "impute" selected. The user selects "mode" as handling strategy instead of "most frequent" (what SimpleImputer expects). I changed it to "mode" to be more understandable as a choice. 
     - BUT: I may want to change this back because the agent will suggest "most_frequent" (since this is how it is ultimately encoded in the dictionary).

3. Preprocess data

    a) handle_dtypes
    - missing_value_precheck:
      - `TODO`: Add more possible missing value markers that could be detected.
      - If more than 1 missing value is found:
        - `TODO`: Implement option to replace only specific markers out of the detected list.
        - `TODO`: Implement option to view col distribution for each marker (selecting one at a time (maybe even multiple)).
      - If only one missing value is found:
        - `TODO`: Add an option to view the column distribution for the marker.
      - possible `TODO`: Incorporate my previous code such that missing values that aren't included in my list can be manually added by the user (would likely have to include a dtype specification just in case).
    - handle_semantic_types:
      - `TODO`: Include an option for the agent to "auto complete" types based on the data cols and characteristics.
        - Requires including a confirmation of the semantic types and possibly granting the user the change to manually and/or agentically correct the data types specified for the dataset.
      - `TODO`: Add a way to select multiple columns (similar to proposed handle_dtypes and data_scope selection).
  
    b) select_generator
    - `TODO`: Add parameter options for the data generators and and instruct the agent to give recommendations for generator settings.
     - If generator is already selected:
     - Auto:
       - `TODO`: If "edit" is selected, the user still can get the model's input (pros/cons) on the decision instead of immediately running the tool.
       - `TODO`: If "respond" is selected case needs fixing. If user adds a response, no model is suggested and the process stops running. Check if I forgot to connect something.
    
    c) handle_missing_values
    - Threshold mode: (first check if needed)
      - `TODO`: Add a limit so that the threshold cannot be more than the number of columns.
      - possible `TODO`: Allow for threshold changes if the current threshold was found to be too harsh.
      - TODO: Fix if threshold changes n and auto selected.

4. Save information for synthetic generator tool use:

    a) Save prepared dataset

    b) Make JSON for the dataframe

5. Generate dataset_report from the cleared data: (NOT YET IMPLEMENTED)
   - `TODO`: Look into underlying correlations that could impact the dataset.
     - This might prompt removing a column (or +), so maybe what could be done is make the JSON after this step instead.
   - `Note`: there is no formal dataset_report as of yet (just information compiled)
     - Information (generator selection): Considers synthesis request and semantic types.
   - possible `TODO`: a more fleshed out dataset report could be more informative for decisions.

#### Generate synthetic data:
6. Run the selected data generator with desired sample number/count

7. Evaluate metrics comparing generated data with cleaned dataset
     - `TODO`: Incorporate more extensive evaluation with Syndat (more metrics etc.)
     - Can graphs be added?
     - TODO: Instruct agent to consider information given about the dataset to reflect on synthesis choices (e.g., using a generator that is specialized to numerical data gen on a mostly categorical dataset -> Observing this and changing the plan given past choices)
       - Give it more information about how it can update the plan.

8. Agent uses metrics along with dataset information and synthesis goal to create an evaluation report. (summarize findings) (not yet implemented)
- If the result is unsatisfactory (low metrics), the agent can reflect on alternative paths using the synthesis request for reference to suggest alternative options (e.g., attempting with another generator) and returning to that specific point/block.
  - `TODO`: Add more instruction for the agent to be able to know more about which steps can be revisited to use in the workflow process.
  - `TODO`: Implement a way for the agent to revisit steps that might improve the results.
  - Possible `TODO`: incorporate downstream analysis information (previously predictive_utility).


---

Using generative models from [Synthetic Data Release](https://github.com/spring-epfl/synthetic_data_release).

---
Currently not used in the workflow:

#### Notes from workflow `TODOs`:
* `select_preferences`:    
  - possible `TODO`: Clarify possible confusion if mode "impute" selected. The user selects "mode" as handling strategy instead of "most frequent" (what SimpleImputer expects). I changed it to "mode" to be more understandable as a choice. 
     - BUT: I may want to change this back because the agent will suggest "most_frequent" (since this is how it is ultimately encoded in the dictionary).
* `check_dtypes`: Add features (described below) to missing_value_precheck (view of col occurrence prior to changing to NA).

3 main blocks: Generate dataset report, recommend synthetic generation plan, and generate data and report output.

1. Generate dataset_report from the cleared data: (NOT YET IMPLEMENTED)
2. Agent uses metrics along with dataset information and synthesis goal to create an evaluation report. (summarize findings) (not yet implemented)
---

`NOTE` on versioning (as discussed): Possibly expand on dataset versions by making a History class which takes the dataframe, the step of when it was updated, and even some context from this (e.g., processing decision made (which imputer, which generator, etc.)).

* The data generators cannot run if the data has NaN values. I give an option `missing value handling` -> `set_threshold` of allowed missing values in a sample, but in the context of using the tools no NaN values are allowed.
  * Because of this, I am not certain whether setting a threshold would be a good option to keep or not. One possibility I am thinking about is that if a threshold is set and missing values still remain, those missing values should be imputed (so, if the user selected "threshold", they would also need to specify their preferred imputation methods just in case).
  * `NOTE`: Possibly implement later, right now just assuming that the agent will be the main one guiding the process.

#### Ignore for now (not planning to implement, left here just in case)
- `TODO`: data_scope, Selecting specific columns to use is not yet implemented. Have to figure out a way to do this without being frustrating.
  - Possibly give the option (keep|exclude|all), but in the first two cases, still manually writing the column names in terminal might be annoying.
- handle_dtypes:
  - possible TODO: Add a way to select multiple columns and convert them to a specific datatype all together (e.g., select columns: 1-2,5,9-15; select data type: int)
- run synthesis
  - If generator is already selected:
     - possible `TODO`: Add a case where the generator is already selected by the user with no agent connection (just directly running the tool).
If necessary:
- Implement more semantic dtypes (`Note`: The tools seem to just have the 4 mentioned, it might not support others (not yet tested)).

note for missing_value_handling:       
- General idea: Agent recommends a plan and if agreed upon, the missing data handling plan is updated to reflect those changes.
        - Upon disagreement, the user is able to write a message explaining what they disagree with and communicate with the agent until a decision has been reached the user agrees with.
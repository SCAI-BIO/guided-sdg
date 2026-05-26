
# agentic-synth-data
Agentic workflow for synthetic data generation &amp; evaluation.

## Update notes:
### From previous meeting:
- [x] Rename upload to load_dataframe and upload_and_preprocess to preprocess
- [x] Include typing (and including returns) to all existing functions
- [x] Automate semantic type guess based on: column name, dtype, and number of unique values
- [x] Commented out the synthesis request preference selection (no "predictive_utility" so early)
- [x] Look into changeable parameters (and what the defaults are for the tools).
  - `NOTE:` The tools have different parameters. Have to think about how an agent would be able to recommend certain params over others (even the default). Maybe start with default and update if changes are needed (think)?
- [x] Do not compare the cleaned dataset (imputation) to the synthetic dataset (switch it to comparing the original dataset).
  - Essentially, the cleaned dataset is disregarded later (removal of missing values and imputation).
  - Q: We mentioned that items meant to represent missing values, should be saved (as they were initially intended to be missing, but written incorrectly).
- [x] Play with the system prompt (e.g., instruct the agent to be more critical of choices made in the synthesis_request and suggested changes. Warn how changes that don't necessarily make sense)
  - `NOTE`: Needs work.
  - [x] Missing value prompt: Add a "warning" section for invalid or nonsensical suggestions from the user and explain the rationale for the warning.
    - [ ] Check: Adding framing of the dataset into the agent's decision.
  - [x] Generator prompt: ask for a structured output (pseudo) and gather (generator, pros, cons, rationale given the dataset, etc.)
  - [ ] Evaluation prompt: give more context on the workflow (limit to something like "Currently, the only change you can call is returning to the generation step adn choosing a new generator. Do this if you see it is needed. The user will approve if the decision is warranted".)


### Additional updates:
## My own checklist:
- [x] Move auto to the top for missing value and generator preference selection pyip menus.
- [x] Make load dataframe the first step in the workflow.
- [x] Fix "auto" selection for num/cat handling. If selected, the agent is not included, it just tries to run SimpleImputer("auto"), which evidently fails.
- [x] Print clear specifications of which stage the user is in (stage name is sufficient for now).
- [x] handle_semantic_types: Remove updates to the dataframe based on semantic type of the data. The semantic types dict is already sufficient in this case as a reference for downstream steps.
- [x] Add dataframe storage to DataGen (e.g., self.synthetic_df, self.handled_dtypes_df, etc.).
  - As discussed for versioning: Possibly expand on dataset versions by making a History class which takes the dataframe, the step of when it was updated, and even some context from this (e.g., processing decision made (which imputer, which generator, etc.)).
- [ ] When prompting for a structured dictionary output, the agent can mess up, which results in a JSON decoding error. Give the agent a chance to reformat its response (would have to give it feedback like "the JSON was formatted incorrectly, refer to the instructions and try again").
  - [ ] There is a way for agents to try again with tools, for this too.
- [ ] Preprocessing `TODO`: Add messages indicating the the df was updated (save previous version).


## Data synthesis workflow (plan):
3 main blocks: Generate dataset report, recommend synthetic generation plan, and generate data and report output.

#### Generate dataset report:
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
    
    b) handle_missing_values
    - Threshold mode:
      - `TODO`: Add a limit so that the threshold cannot be more than the number of columns.
      - possible `TODO`: Allow for threshold changes if the current threshold was found to be too harsh.
      - TODO: Fix if threshold changes n and auto selected.
    - Auto mode:
    - `TODO`: Implement agent pushback (aka warning) based on the dataset's characteristics (changes are still made at the user's whims and currently the agent won't really give pushback for decisions that don't make much sense).
      - possible `TODO`: Allow for more conversation between the agent and user before immediately recommending a plan change?
      - `TODO`: specify to the agent that threshold needs to be an int.

4. Make JSON for the dataframe (required for the data generators to run)

5. Generate dataset_report from the cleared data: (NOT YET IMPLEMENTED)
   - `TODO`: Look into underlying correlations that could impact the dataset.
     - This might prompt removing a column (or +), so maybe what could be done is make the JSON after this step instead.
   - `Note`: there is no formal dataset_report as of yet (just information compiled)
     - Information (generator selection): Considers synthesis request and semantic types.
   - possible `TODO`: a more fleshed out dataset report could be more informative for decisions.

#### Recommend synthetic generation plan:
6. Agent data generator recommendation (takes in dataset_report and synthesis_report) 
    - `TODO`: Add parameter options for the data generators and and instruct the agent to give recommendations for generator settings.
      - `TODO`: Agent does not push back against changes that do not make sense

7. The user validates the recommendation.
   - run_synthesis:
     - If generator is already selected:
       - possible `TODO`: Incorporate a chance for agent feedback (pros/cons) regarding the generator selected, and give the option for the user to still update the generator or get a recommendation.
     - Auto:
       - `TODO`: If "edit" is selected, the user still can get the model's input (pros/cons) on the decision instead of immediately running the tool.
       - `TODO`: If "respond" is selected case needs fixing. If user adds a response, no model is suggested and the process stops running. Check if I forgot to connect something.

#### Generate data and report output:
8. Run the selected data generator with desired sample number/count

9. Evaluate metrics comparing generated data with cleaned dataset
     - `TODO`: Incorporate more extensive evaluation with Syndat (more metrics etc.)
     - Can graphs be added?
     - TODO: Instruct agent to consider information given about the dataset to reflect on synthesis choices (e.g., using a generator that is specialized to numerical data gen on a mostly categorical dataset -> Observing this and changing the plan given past choices)
       - Give it more information about how it can update the plan.

10. Agent uses metrics along with dataset information and synthesis goal to create an evaluation report. (summarize findings)
- If the result is unsatisfactory (low metrics), the agent can reflect on alternative paths using the synthesis request for reference to suggest alternative options (e.g., attempting with another generator) and returning to that specific point/block.
  - `TODO`: Add more instruction for the agent to be able to know more about which steps can be revisited to use in the workflow process.
  - `TODO`: Implement a way for the agent to revisit steps that might improve the results.
  - Possible `TODO`: incorporate downstream analysis information (previously predictive_utility).

---

Using generative models from [Synthetic Data Release](https://github.com/spring-epfl/synthetic_data_release).

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

# agentic-synth-data
Agentic workflow for synthetic data generation &amp; evaluation.

## Update notes:
- Added a more OOP based approach (for the most part retaining the original functions).
  - A lot of time was spent separating and formatting the code into a usable workflow to fit this format. 
    - `Note`: The prototype design notebook did not have a persistent agent in the sense that more than one agent was created to facilitate the workflow in the notebook (and there was an instance where the model was directly called instead of the agent), so that had to be changed for this new approach.

# Versioning (proposed):
 * `TODO`: Add a save to retry from a previous step (saving the dataframe, possibly additional details as well) so steps can be more fluidly revisited.
   * Possible implementation: dictionary of lists where keys are steps and the most recent entries added to the end of the lists (but maybe even then, it would be helpful to have a note of what was done for reference).
   * Incorporate a save for each version of the dataset (trailing v1,2,etc. for each df stage if revisited)

## Data synthesis workflow (plan):
3 main blocks: Generate dataset report, recommend synthetic generation plan, and generate data and report output.

#### Generate dataset report:
1. Select preferences
   - `TODO`: Missing integration for predictive utility consideration.
   - `TODO`:Selecting "auto" for numerical or categorical handling is not yet supported (need to connect to agent if auto in any of these are selected).
   - possible `TODO`: Setup the dictionary in the nested format (not consistent in preference selection). 
   - `TODO`: data_scope, Selecting specific columns to use is not yet implemented. Have to figure out a way to do this without being frustrating.
     - Possibly give the option (keep|exclude|all), but in the first two cases, still manually writing the column names in terminal might be annoying.
   - possible `TODO`: For the imputer selection, I changed the setting the user selects to be "mode" instead of "most frequent", I may want to change this back as in missing value handling, the agent will suggest "most_frequent" (since this is how it is ultimately encoded in the dictionary).

2. Upload data (verify valid format for pandas)
   - `TODO`: Add a dataset size minimum (the tools need about 150 samples, if I am remembering right).

3. Clean deterministically
   * `TODO`: Add messages indicating the the df was updated (save previous version).
  
    a) handle_dtypes
    - missing_value_precheck:
      - `TODO`: Add more possible missing value markers that could be detected.
      - If more than 1 missing value is found:
        - `TODO`: Implement option to replace only specific markers out of the detected list.
        - `TODO`: Implement option to view col distribution for each marker (selecting one at a time (maybe even multiple)).
      - If only one missing value is found:
        - `TODO`: Add an option to view the column distribution for the marker.
      - possible `TODO`: Incorporate my previous code such that missing values that aren't included in my list can be manually added by the user (would likely have to include a dtype specification just in case).
    - handle_dtypes:
      - possible TODO: Add a way to select multiple columns and convert them to a specific datatype all together (e.g., select columns: 1-2,5,9-15; select data type: int)
    - handle_semantic_types:
      - `TODO`: Include an option for the agent to "auto complete" types based on the data cols and characteristics.
        - Requires including a confirmation of the semantic types and possibly granting the user the change to manually and/or agentically correct the data types specified for the dataset.
      - `TODO`: Add a way to select multiple columns (similar to proposed handle_dtypes and data_scope selection).
    
    b) handle_missing_values
    - Threshold mode:
      - `TODO`: Add a limit so that the threshold cannot be more than the number of columns.
      - possible TODO: Allow for threshold changes if the current threshold was found to be too harsh.
    - Auto mode:
      - `TODO`: Fix recommendation numerical/categorical_handling when "auto" is selected. Currently, no recommendation is made at all.
    - `TODO`: Implement agent pushback (aka warning) based on the dataset's characteristics (changes are still made at the user's whims and currently the agent won't really give pushback for decisions that don't make much sense).
      - possible `TODO`: Allow for more conversation between the agent and user before immediately recommending a plan change?

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
       - possible `TODO`: Add a case where the generator is already selected by the user with no agent connection (just directly running the tool).
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

10. Agent uses metrics along with dataset information and synthesis goal to create an evaluation report. (summarize findings)
- If the result is unsatisfactory (low metrics), the agent can reflect on alternative paths using the synthesis request for reference to suggest alternative options (e.g., attempting with another generator) and returning to that specific point/block.
  - `TODO`: Add more instruction for the agent to be able to know more about which steps can be revisited to use in the workflow process.
  - `TODO`: Implement a way for the agent to revisit steps that might improve the results.

---
#### Ignore for now:
If necessary:
- Implement more semantic dtypes (`Note`: The tools seem to just have the 4 mentioned, it might not support others (not yet tested)).

note for missing_value_handling:       
- General idea: Agent recommends a plan and if agreed upon, the missing data handling plan is updated to reflect those changes.
        - Upon disagreement, the user is able to write a message explaining what they disagree with and communicate with the agent until a decision has been reached the user agrees with.
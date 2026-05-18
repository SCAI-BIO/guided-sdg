
# agentic-synth-data
Agentic workflow for synthetic data generation &amp; evaluation.

## Update notes:
- Added a more OOP based approach (for the most part retaining the original functions).
  - A lot of time was spent separating and formatting the code into a usable workflow to fit this format. 
    - `Note`: The prototype design notebook did not have a persistent agent in the sense that more than one agent was created to facilitate the workflow in the notebook (and there was an instance where the model was directly called instead of the agent), so that had to be changed for this new approach.

## Data synthesis workflow (plan):

3 main blocks: Generate dataset report, recommend synthetic generation plan, and generate data and report output.

#### Generate dataset report:
1. Upload data (verify valid format for pandas)
   - TODO: in selecting preferences, add the chance to select specific columns in a dataset (include or exclude them prior to 'treatment').
2. Clean deterministically
    
    a) handle_dtypes
    - `TODO`: Add missing value heuristic (automatic detection).
      - Still would want to include a chance to include non standard missing value symbols that might be necessary (think how).
    - `TODO`: In the semantic datatype handling: Include an option for an agent to "auto complete" based on the data cols and characteristics what type of semantic data type it belongs to.
      - This would include a confirmation of the semantic types and possibly a chance for the user to manually and/or agentically correct the data types specified for the dataset.
    
    b) handle_missing_values
    - `TODO`: Fix recommendation numerical/categorical_handling when "auto" is selected. Currently, no recommendation is made at all.
    - `TODO`: Changes are still made at the user's whims and the agent won't really give pushback for decisions that don't make much sense in the analysis.
      - possible `TODO`: Allow for more conversation between the agent and user before immediately recommending a plan change?
      * Need to confirm whether the agent still recommends non valid entries (e.g. None for numerical handling)
3. Make JSON for the dataframe (required for the data generators to run)
4. Generate dataset_report from the cleared data
   - `TODO`: Look into underlying correlations that could impact the dataset (this might end up being done before step `3`).
   - `Note`: there is no formal dataset_report as of yet (just information compiled)
   - possible TODO: a more fleshed out dataset report could be more informative for decisions.

#### Recommend synthetic generation plan:
5. Agent data generator recommendation (takes in dataset_report and synthesis_report) 
    - `TODO`: Add parameter options for the data generators and and instruct the agent to give recommendations for generator settings.
      - `TODO`: Agent does not push back against changes that do not make sense
6. The user validates the recommendation.

#### Generate data and report output:
7. Run the selected data generator with desired sample number/count
8. Evaluate metrics comparing generated data with cleaned dataset
     a) `TODO`: Incorporate more extensive evaluation with Syndat (more metrics etc.)
     - Keep in mind the general semantic types present (there was a tool that didn't work on categorical data, so mostly categorical datasets would probably not benefit from the tool)
9. Agent uses metrics along with dataset information and synthesis goal to create an evaluation report. (summarize findings)
- If the result is unsatisfactory (low metrics), the agent can reflect on alternative paths using the synthesis request for reference to suggest alternative options (e.g., attempting with another generator) and returning to that specific point/block.
  - `TODO`: Add more instruction for the agent to be able to know more about which steps can be revisited to use in the workflow process.
  - `TODO`: Implement a way for the agent to revisit steps that might improve the results.
  - `TODO`: Incorporate a save for each version of the dataset (trailing v1,2,etc. for each df stage if revisited)

## Goals
Tasks:
- Try to get project so far to a point that it can run consistently.
- OOP approach to implementation.
- More object oriented approach [not as important for now]

If necessary:
- Implement more semantic dtypes (`Note`: The tools seem to just have the 4 mentioned, it might not support others (not yet tested)).

---

note for missing_value_handling:       
- General idea: Agent recommends a plan and if agreed upon, the missing data handling plan is updated to reflect those changes.
        - Upon disagreement, the user is able to write a message explaining what they disagree with and communicate with the agent until a decision has been reached the user agrees with.
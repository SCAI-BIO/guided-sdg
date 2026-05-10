
# agentic-synth-data
Agentic workflow for synthetic data generation &amp; evaluation.

## Data synthesis workflow (plan):

3 main blocks: Generate dataset report, recommend synthetic generation plan, and generate data and report output.

#### Generate dataset report:
1. Upload data (verify valid format for pandas)
2. Clean deterministically
    
    a) handle_dtypes
    - TODO: Incorporate additional semantic types (I have only included the ones the tools showed as useable in the demo)
    
    b) handle_missing_values
    - Note: if "auto" is selected the current missing value plan with dataset information is handed to the agent for it to decide which settings to select.
      - General idea: Agent recommends a plan and if agreed upon, the missing data handling plan is updated to reflect those changes.
        - Upon disagreement, the user is able to write a message explaining what they disagree with and communicate with the agent until a decision has been reached the user agrees with.
        - possible TODO: Changes are made still at the user's whims and is not very conversational/possibly too close to initial reasoning message. Maybe allow for more conversation before immediately recommending a plan change?
3. Make JSON for the dataframe (required for the data generators to run)
4. Generate dataset_report from the cleared data
   - possible TODO: a more fleshed out dataset report could be more informative for decisions.

#### Recommend synthetic generation plan:

5. Agent data generator recommendation (takes in dataset_report and synthesis_report)
    - TODO: If the user wants changes, allow a way for changes to be enacted, communicating with the agent to format everything for the data generators.
6. The user validates the recommendation.
   - TODO: Add this feature before generating the synthetic data.

#### Generate data and report output:
TODO: Still need to implement this section

7. Run the selected data generator with desired sample number/count
8. Evaluate metrics comparing generated data with cleaned dataset
     a) syndat.
9. Agent uses metrics along with dataset information and synthesis goal to create an evaluation report. (summarize findings)
- If the result is unsatisfactory (low metrics), the agent can reflect on alternative paths using the synthesis request for reference to suggest alternative options (e.g., attempting with another generator) and returning to that specific point/block.

## Goals
Tasks:
- Incorporate Syndat eval.
- Try to get project so far to a point that it can run consistently.
- OOP approach to implementation.
- More object oriented approach [not as important for now]

Want to incorporate:
- Dataset versions (trailing v1,2,etc.) for tracking dfs.
- More semantic dtypes (if accepted)
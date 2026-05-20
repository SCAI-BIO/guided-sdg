import pandas as pd
import numpy as np
import pyinputplus as pyip
import pprint
import json
from typing import Optional
from langgraph.graph.state import CompiledStateGraph

from sklearn.impute import SimpleImputer

from agentic_synth_data.dataset_preparation.semantic_type_handling import match_to_semantic_types
from agentic_synth_data.utils import format_response
from agentic_synth_data.prompts.missing_value_prompt import missing_value_prompt #, MISSING_VALUE_SYSTEM_PROMPT


def update_missing_value_choice(synthesis_request: dict):
    print("Please select another method to handle missing values:")

    choice = pyip.inputChoice(["impute", "threshold", "auto"])
    synthesis_request["missing_value_handling"]["mode"] = choice

    # If any additional specifications are required
    if choice == "threshold":
        value = pyip.inputInt("Set the maximum number of missing values allowed per row:", min=0)
        synthesis_request["missing_value_handling"][choice]["set_threshold"] = value

    elif choice == "impute":
        #method = pyip.inputChoice(["NA"], prompt="Please select an imputation method:")
        synthesis_request["missing_value_handling"][choice]["method"] = "SimpleImputer"
        print("\nHow would you like to handle missing categorical values? Select 'auto' if you would like a suggestion.")
        choice = pyip.inputMenu(["auto", "mode", "constant"], numbered=True)
        if choice == "mode": # Changed externally to be a bit more understandable: mode vs most_frequent
            synthesis_request["missing_value_handling"]["impute"]["categorical_strategy"] = "most_frequent"
        synthesis_request["missing_value_handling"]["impute"]["categorical_strategy"] = choice

        print("\nHow would you like to handle missing numerical values? Select 'auto' if you would like a suggestion.")
        choice = pyip.inputMenu(["auto", "mean", "median", "mode"], numbered=True)
        if choice == "mode":
            synthesis_request["missing_value_handling"]["impute"]["numerical_strategy"] = "most_frequent"
        synthesis_request["missing_value_handling"]["impute"]["numerical_strategy"] = choice

    return synthesis_request


def update_missing_handling_plan(synthesis_request:dict, missing_plan:dict):
    update = synthesis_request["missing_value_handling"].copy()

    update["mode"] = missing_plan["mode"]

    update["impute"]["method"] = missing_plan["imputation_method"]
    update["impute"]["categorical_strategy"] = missing_plan["categorical_strategy"]
    update["impute"]["numerical_strategy"] = missing_plan["numerical_strategy"]

    update["threshold"]["set_threshold"] = missing_plan["set_threshold"]

    synthesis_request["missing_value_handling"] = update

    return synthesis_request

def handle_missing_values(df:pd.DataFrame, synthesis_request:dict, semantic_types:dict, agent:Optional[CompiledStateGraph] | None = None, config:dict | None = None):
    if df.isna().sum().sum() == 0:
        print("There are no missing values in the dataset.")
    else:
        #TODO add an initial check for missing values
        mode = synthesis_request["missing_value_handling"]["mode"]
        # 1. Remove all samples with NA columns
        if mode == "remove":
            null_row_count = df.isna().any(axis=1).sum()
            print(f"Warning: {null_row_count} out of {df.shape[0]} rows ({null_row_count/df.shape[0]*100}%) will be removed from the dataset.")
            confirm = pyip.inputYesNo("Confirm whether this should be done (yes/no/y/n):")
            if confirm == "yes":
                df = df.dropna()
                return df, synthesis_request
            else:
                # If this option is too harsh and the user wants to try another method.
                synthesis_request = update_missing_value_choice(synthesis_request)
                return handle_missing_values(df, synthesis_request, semantic_types)
            
        # 2. Remove all samples above a specified threshold of NA columns 
        elif mode == "threshold":
            threshold = synthesis_request["missing_value_handling"]["threshold"]["set_threshold"]
            # TODO add a limit to the threshold such that it doesn't add mroe than
            threshold_samples = df.isna().sum(axis=1) > threshold
            count = threshold_samples.sum()

            print(f"Warning: {count} out of {df.shape[0]} ({count/df.shape[0]*100}%) will be removed from the dataset.")
            confirm = pyip.inputYesNo("Confirm whether this should be done.")
            if confirm == "yes":
                df = df[~threshold_samples]
                return df, synthesis_request
            else:
                confirm = pyip.inputYesNo("Would you like to select a new threshold value?")
                if confirm == "yes":
                    value = pyip.inputInt("Enter a new threshold value.", min=1) # Double check how I have it written
                    synthesis_request["missing_value_handling"]["set_threshold"] = value
                    return handle_missing_values(df, synthesis_request, semantic_types)
                    
                synthesis_request = update_missing_value_choice(synthesis_request)
                # TODO add a limit to the threshold such that it doesn't add more
                #  possible TODO allow for threshold changes (if found to be too harsh but want to stay with a threshold)

                return handle_missing_values(df, synthesis_request, semantic_types)

        elif mode == "impute":
            # Separate user defined categorical and numerical values for imputation
            cat_cols = df.select_dtypes(include="category").columns.to_list()
            num_cols = df.select_dtypes(include="number").columns.to_list()

            print("\nReview of missing value handling results:")
            # Categorical value imputation
            cat_strategy = synthesis_request["missing_value_handling"]["impute"]["categorical_strategy"]
            if cat_cols:
                cat_data = df[cat_cols].astype("object").where(df[cat_cols].notna(), np.nan)
                if cat_strategy == "constant":
                    print("\nEnter the fill value that missing values should have. Fill values must be strings.\nPress enter if you would like to stay with the default ('Unknown'):")
                    fill_value = pyip.inputStr()
                    if fill_value == "":
                        fill_value = "Unknown"
                        imputer = SimpleImputer(strategy=cat_strategy, fill_value=fill_value)
                elif cat_strategy == "mode":
                    imputer = SimpleImputer(strategy="most_frequent")
                else:
                    imputer = SimpleImputer(strategy=cat_strategy)

                df[cat_cols] = pd.DataFrame(imputer.fit_transform(cat_data),columns=cat_cols, index=df.index)
                print(f"- There are {df[cat_cols].isna().sum().sum()} remaining null values in the categorical columns.")

            # Numerical value imputation
            if num_cols:
                num_strategy = synthesis_request["missing_value_handling"]["impute"]["numerical_strategy"]
                imputer = SimpleImputer(strategy=num_strategy)
                df[num_cols] = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=df[num_cols].columns, index=df.index)
                print(f"- There are {df[num_cols].isna().sum().sum()} remaining null values in the numerical columns.")

            print(f"- There are {df.isna().sum().sum()} remaining null values in the dataset.")
            df = match_to_semantic_types(df, semantic_types)
            return df, synthesis_request

        # 5. Use an LLM to suggest the missing value handling strategy
        if mode == "auto":
            proportion_missing = (df.isna().sum() / df.shape[0]).round(4).to_dict()
            rows_with_missing = int(df.isna().any(axis=1).sum())
            total_rows = df.shape[0]
            plan_update = None

            prompt = missing_value_prompt(proportion_missing=proportion_missing, rows_with_missing=rows_with_missing, total_rows=total_rows, synthesis_request=synthesis_request, semantic_types=semantic_types, plan_update=plan_update)

            feedback = ""
            while True:
                if feedback is None:
                    current_prompt = prompt
                else:
                    current_prompt = f"""
{prompt}

The user rejected the previous plan and gave this feedback:
{feedback}

Revise the missing-value handling plan according to this feedback.
Return exactly one valid JSON object.
"""
                plan = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": current_prompt,
                            }
                        ]
                    },
                    config=config,
                )
                plan = plan['messages'][-1].content.split("</think>")[1].replace("\n", "")
                parsed = json.loads(plan)
                missing_plan = parsed["missing_value_handling"]

                print("\nSuggested plan:")
                print(f"- Mode: {missing_plan['mode']}")
                print(f"- Threshold: {missing_plan['set_threshold']}")
                print(f"- Imputation method: {missing_plan['imputation_method']}")
                print(f"- Categorical strategy: {missing_plan['categorical_strategy']}")
                print(f"- Numerical strategy: {missing_plan['numerical_strategy']}")

                if len(missing_plan["columns_to_remove"]) > 0:
                    print(f"- Columns to remove: {missing_plan["columns_to_remove"]}")
                
                print(f"\nReasoning behind the suggestion:")
                format_response(missing_plan["rationale"])

                confirm = pyip.inputYesNo("\nWould you like to proceed with the suggestion?")
                if confirm == "yes":
                    synthesis_request.update({
                        "mode": missing_plan["mode"],
                        "set_threshold": missing_plan["set_threshold"],
                        "imputation_method": missing_plan["imputation_method"],
                        "categorical_strategy": missing_plan["categorical_strategy"],
                        "numerical_strategy": missing_plan["numerical_strategy"],
                        "columns_to_remove": missing_plan["columns_to_remove"],
                        })
                    print(f"\n\nThe missing value handling plan has been updated to the following:")
                    pprint.pprint(synthesis_request)
                    break

                else:
                    print("\nPlease describe how you would want missing values to be handled and what changes you would make to the suggested plan.")
                    feedback = input()

                print(f"\nReasoning behind the suggestion:")
                print(format_response(missing_plan["rationale"]))

            # Update synthesis request with new specifications
            synthesis_request = update_missing_handling_plan(synthesis_request, missing_plan)
            return handle_missing_values(df, synthesis_request, semantic_types)
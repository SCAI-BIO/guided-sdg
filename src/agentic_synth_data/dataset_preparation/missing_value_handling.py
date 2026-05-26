import pandas as pd
import numpy as np
import pyinputplus as pyip
import pprint
import json
from typing import Optional, Any
from langgraph.graph.state import CompiledStateGraph

from sklearn.impute import SimpleImputer

from agentic_synth_data.utils import format_response
from agentic_synth_data.prompts.missing_value_prompt import missing_value_prompt, missing_feedback


def update_missing_value_choice(synthesis_request:dict[str, Any]) -> dict[str, Any]:
    """Update missing value handling choice in the case that either the remove all or
     threshold setting for null values is too harsh.

    Args:
        synthesis_request (dict[str, Any]): User preferences for data synthesis.

    Returns:
        dict[str, Any]: Updated synthesis request.
    """
    print("Please select another method to handle missing values:")

    choice = pyip.inputMenu(["auto", "impute", "threshold",], numbered=True)
    synthesis_request["missing_value_handling"]["mode"] = choice

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


def auto_update_missing_plan(synthesis_request:dict[str, Any], missing_plan:dict[str, Any]) -> dict[str, Any]:
    """When "auto" is selected in the missing value handling plan, the agent is called and 
    returns a structured dictionary. 

    Args:
        synthesis_request (dict): User preferences for data synthesis.
        missing_plan (dict): Agent recommendation for handling missing values.

    Returns:
        dict: Updated synthesis request.
    """
    update = synthesis_request["missing_value_handling"].copy()

    update["mode"] = missing_plan["mode"]

    update["impute"]["method"] = missing_plan["imputation_method"]
    update["impute"]["categorical_strategy"] = missing_plan["categorical_strategy"]
    update["impute"]["numerical_strategy"] = missing_plan["numerical_strategy"]

    update["threshold"]["set_threshold"] = missing_plan["set_threshold"]

    synthesis_request["missing_value_handling"] = update

    return synthesis_request


def handle_missing_values(
        df:pd.DataFrame, 
        synthesis_request:dict[str, Any], 
        semantic_types:dict[str, str], 
        agent:Optional[CompiledStateGraph] | None = None, 
        config:dict | None = None,
        ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Handle missing values in a dataframe based on user defined preferences.

    Missing value strategies include:
    1. Remove all rows containing null values.
    2. Remove rows exceeding a specified threshold for null values.
    3. Imputation.
    4. Receive a suggestion from an agent on how to handle null values.

    Args:
        df (pd.DataFrame): The input dataframe.
        synthesis_request (dict[str, Any]): User preferences for data synthesis.
        semantic_types (dict[str, str]): Map of column names to semantic types (e.g., categorical, 
                               ordinal, integer, float).
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to 
                               recommend a strategy for missing values. Defaults to None.
        config (dict | None, optional): Configuration passed to the agent during invocation. Defaults to None.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]: 
        - (pd.DataFrame) The processed dataframe after missing value handling. 
        - (dict[str, Any]) The updated synthesis_request.
    """
    if df.isna().sum().sum() == 0:
        print("\nThere are no missing values in the dataset. Continuing to the next step.")
        return df, synthesis_request
    else:
        print(f"\n{df.isna().sum().sum()} missing values detected in the dataset.")
        mode = synthesis_request["missing_value_handling"]["mode"]
        impute = synthesis_request["missing_value_handling"]["impute"]
        
        # 1. Remove all samples with NA columns.
        if mode == "remove":
            null_row_count = df.isna().any(axis=1).sum()
            print(f"\nWarning: {null_row_count} out of {df.shape[0]} rows ({null_row_count/df.shape[0]*100}%) will be removed from the dataset.")
            confirm = pyip.inputYesNo("Confirm whether this should be done (yes/no/y/n):")
            if confirm == "yes":
                df = df.dropna()
                return df, synthesis_request
            else:
                # If this option is too harsh and the user wants to try another method.
                synthesis_request = update_missing_value_choice(synthesis_request)
                return handle_missing_values(df, synthesis_request, semantic_types)
            
        # 2. Remove all samples above a specified threshold of NA columns. 
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
                confirm = pyip.inputYesNo("\nWould you like to select a new threshold value?")
                if confirm == "yes":
                    value = pyip.inputInt("\nEnter a new threshold value.", min=1) # Double check how I have it written
                    synthesis_request["missing_value_handling"]["set_threshold"] = value
                    return handle_missing_values(df, synthesis_request, semantic_types)
                else:  
                    synthesis_request = update_missing_value_choice(synthesis_request)
                # TODO add a limit to the threshold such that it doesn't add more
                #  possible TODO allow for threshold changes (if found to be too harsh but want to stay with a threshold)

                return handle_missing_values(df, synthesis_request, semantic_types)

        # 3. Perform imputation given there are decided categorical and numerical strategies.
        elif (
            mode == "impute" and 
            impute["categorical_strategy"] != "auto" and 
            impute["numerical_strategy"] != "auto"
        ):
            # Separate user defined categorical and numerical values for imputation
            grouped = {}
            for col, stype in semantic_types.items():
                grouped.setdefault(stype, []).append(col)

            cat_cols = grouped.get("Categorical", []) + grouped.get("Ordinal", [])
            num_cols = grouped.get("Integer", []) + grouped.get("Float", [])

            df = df.copy()
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
                    print(f"Missing values filled with '{fill_value}'")
                
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

            print(f"\n=> There are {df.isna().sum().sum()} remaining null values in the dataset.")
            return df, synthesis_request

        # 4. Use the agent to suggest the missing value handling strategy.
        if (
            mode == "auto" or 
            impute["categorical_strategy"] == "auto" or 
            impute["numerical_strategy"] == "auto"
        ):
            proportion_missing = (df.isna().sum() / df.shape[0]).round(4).to_dict()
            rows_with_missing = int(df.isna().any(axis=1).sum())
            total_rows = df.shape[0]

            prompt = missing_value_prompt(
                proportion_missing=proportion_missing, 
                rows_with_missing=rows_with_missing, 
                total_rows=total_rows, 
                synthesis_request=synthesis_request, 
                semantic_types=semantic_types, 
                )
            print("\nAgent is coming up with a suggestion for missing value handling...")
            feedback = None
            while True:
                if feedback is None:
                    current_prompt = prompt
                else:
                    current_prompt = missing_feedback(prompt, feedback)
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
                #print(plan) # sometimes agent outputs incorrect JSON format
                missing_plan = json.loads(plan)

                print("\nSuggested plan:")
                print(f"- Mode: {missing_plan['mode']}")
                print(f"- Threshold: {missing_plan['set_threshold']}")
                print(f"- Imputation method: {missing_plan['imputation_method']}")
                print(f"- Categorical strategy: {missing_plan['categorical_strategy']}")
                print(f"- Numerical strategy: {missing_plan['numerical_strategy']}")

                if len(missing_plan["columns_to_remove"]) > 0:
                    print(f"- Columns to remove: {missing_plan["columns_to_remove"]}")
                    # TODO: Implement a way to remove columns.
                
                if len(missing_plan["warning"]) > 0:
                    print("\nWarning:")
                    format_response(f"- Warning: {missing_plan["warning"][0]}")
                
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
                    print("\n\nPlease describe how you would want missing values to be handled and what changes you would make to the suggested plan.")
                    feedback = input()

            # Update synthesis request with new specifications
            synthesis_request = auto_update_missing_plan(synthesis_request, missing_plan)
            return handle_missing_values(df, synthesis_request, semantic_types)
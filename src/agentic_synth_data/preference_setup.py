from typing import Any
import pprint

import pyinputplus as pyip

from agentic_synth_data.utils import format_response

# possible TODO: change all parts to the nested format (have to match with all other files that call the same information)
# TODO: allow for selection of columns of interest

def select_preferences() -> dict[str, Any]:
    """Initial setup of user preferences.

    Returns:
        dict[str, Any]: Dictionary of the user's preferences.
    """
    #cols_of_interest = [] # if empty, all columns should be included
    synthesis_request = {
        # Q: What would you like to use the synthetic data for?
        # "synthesis_goal": {
        #     "goal": "basic_generation",  
        #     # TODO: change predictive utility to the format below
        #     # "predictive_utility": {
        #     #     "task": None, # regression, classification, survival, None
        #     #     "target_column": None,
        #     # }
        #     "task": None,
        #     "target_column": None,
        # },

        # Q: Which columns should be included?
        # Should assume that the user wants to use all columns uploaded.
        # "data_scope": {
        #     "selected_columns": cols_of_interest # If empty, all cols
        #     },
        "sample_size": {
            # possible TODO: add fraction of the original dataset as an option (non-essential)
            "mode": None, # fixed_value, fraction, same_as_original
            "fixed": {
                "value": None,
            },
        },
        "privacy": {
            "mode": None, # standard, differentially private
            # TODO change to this structure
            # "differentially_private": {
            #     "sensitive_columns": [],
            # }
            "sensitive_columns": [],
        },
        "generator": {
            "name": None, # auto, CTGAN, PATEGAN, BaysianNet, PrivBayes
            #"parameters": {}, # Depends on generators used/available (not important for right now)
        },
        "missing_value_handling": {
            "mode": None, # auto, impute, threshold, remove
            "impute": {
                "method": None, # SimpleImputer
                "categorical_strategy": None, # auto, most_frequent, constant
                "numerical_strategy": None, # auto, mean, median, most_frequent
            },
            "threshold": {
                "set_threshold": None, # minimum number of missing columns in a sample to disregard
            }
        }
    }

    # Fill synthesis goal
    # print("What would you like to use the synthetic data for?")
    # choice = pyip.inputMenu(["basic generation", "predictive utility"], numbered=True)
    # if choice == "predictive utility":
    #     synthesis_request["synthesis_goal"]["goal"] = "predictive_utility"
    #     print("\nWhich downstream machine learning task are you interested in?")
    #     choice = pyip.inputMenu(["classification", "regression", "survival"], numbered=True) # default none
    #     synthesis_request["synthesis_goal"]["task"] = choice
    #     print("\nWhat is the target column? Select 'auto' if you would like a suggestion.")
        # TODO think about how to include column names in an easier way
            # I can think of giving an all selection option, but it is very annoying to individually
            # and perfectly type in all column names to be accepted.
        #target = pyip.inputChoice([],"If applicable, what is the target column?") # one entry or auto?
        #synthesis_request["synthesis_goal"]["target_column"] = target 
    # Fill data_scope: Assume user wants to use all of the columns in the df uploaded
    #response = pyip.inputYesNo("Are there any columns in the dataset that you would like to exclude from generation?")
        # Q: Which columns should be included?
        # TODO add above questoin as prompt
    #synthesis_request["data_scope"]["selected_columns"] = cols_of_interest
   
    # Fill sample size
    #print("\nWould you like to generate the same amount of samples as the dataset or a fixed number of samples?")
    format_response("Would you like to generate the same amount of samples as the dataset or a fixed number of samples?")

    choice = pyip.inputMenu(["fixed value", "same as original"], numbered=True) # fixed, same_as_original, fraction_of_original
    synthesis_request["sample_size"]["mode"] = choice
    
    if choice == "fixed value":
        print()
        format_response("How many synthetic samples would you like to generate?")

        value = pyip.inputInt(min=1)
        synthesis_request["sample_size"]["fixed"]["value"] = value
    
    # possible TODO: incorporate an option to for user to input a fraction of the dataset.

    response = pyip.inputYesNo("\nAre you interested in differential privacy?")
    if response == "yes":
        synthesis_request["privacy"]["mode"] = "differential_privacy"
        # TODO: check if there are any sensitive columns (what is a way to include columns that is not cumbesome?)

    # Fill data generator preferences
    print("\n")
    format_response("Is there a specific data generator that you want to use on your data? Select 'auto' if you would like a suggestion.")
    choice = pyip.inputMenu(["auto", "CTGAN", "PATEGAN", "BaysianNet", "PrivBayes"], numbered=True)
    if choice == "auto":
        print("\t=> You will receive a recommendation based on the dataset's characteristics.")
        synthesis_request["generator"] = choice
    else:
        print(f"\t=> Your data will be generated using {choice}.")
    synthesis_request["generator"] = choice
        
    # Missing value handling preferences
    print()
    format_response("How would you like missing values to be handled? Select 'auto' if you would like a suggestion.")
    choice = pyip.inputMenu(["auto", "impute", "threshold", "remove"], numbered=True)
    if choice == "auto":
        synthesis_request["missing_value_handling"]["mode"] = choice
        print("\t=> You will receive a recommendation based on the dataset's characteristics.")

    elif choice == "threshold":
        # TODO: Add threshold limit
        synthesis_request["missing_value_handling"]["mode"] = choice
        value = pyip.inputInt("\nWhat is the minimum number of missing columns in a sample remove from the data?")
        synthesis_request["missing_value_handling"][choice]["set_threshold"] = value

    elif choice == "remove":
        synthesis_request["missing_value_handling"]["mode"] = choice

    elif choice == "impute":
        synthesis_request["missing_value_handling"]["mode"] = choice
        print()
        format_response("Currently, only the SimpleImputer imputation method is currently supported.")
        synthesis_request["missing_value_handling"][choice]["method"] = "SimpleImputer"
        print("\t=> Your missing data will be imputed by SimpleImputer.")
        # TODO Add additional imputers
        
        # Specify categorical handling.
        print()
        format_response("How would you like to handle missing categorical values? Select 'auto' if you would like a suggestion.")
        choice = pyip.inputMenu(["auto", "mode", "constant"], numbered=True)
        if choice == "mode": # Changed externally to be a bit more understandable: mode vs most_frequent
            synthesis_request["missing_value_handling"]["impute"]["categorical_strategy"] = "most_frequent"
        if choice == "auto":
            print("\t=> You will receive a recommendation based on the dataset's characteristics.")
            synthesis_request["generator"] = choice
        else:
            print(f"\t=> Your missing categorical data will be imputed using {choice}.")

        synthesis_request["missing_value_handling"]["impute"]["categorical_strategy"] = choice

        # Specify numerical handling.
        print()
        format_response("How would you like to handle missing numerical values? Select 'auto' if you would like a suggestion.")
        choice = pyip.inputMenu(["auto", "mean", "median", "mode"], numbered=True)
        if choice == "mode":
            synthesis_request["missing_value_handling"]["impute"]["numerical_strategy"] = "most_frequent"
        if choice == "auto":
            print("\t=> You will receive a recommendation based on the dataset's characteristics.")
            synthesis_request["generator"] = choice
        else:
            print(f"\t=> Your missing numerical data will be imputed using {choice}.")
        synthesis_request["missing_value_handling"]["impute"]["numerical_strategy"] = choice
 
    print()
    print("-"*25+"Your preferences have been recorded"+"-"*25)
    pprint.pprint(synthesis_request)
    return synthesis_request
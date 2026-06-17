from typing import Any


def semantic_type_message(column_profiles:dict[str, dict[str, Any]]) -> str:
    return f"""
Current step: Infer semantic types for each column in the dataframe.

Infer the semantic types of each column in the dataframe based on their profiles.
Each column's profile contains the Python data type of the column (str), the value 
counts of unique non null values, and the number of unique non null values.

Note: A float64 column contains at least one value with a decimal component, meaning a non zero decimal exists.

Allowed semantic types:
- Float
- Integer
- Boolean
- Categorical
- Ordinal

Column profiles:
{column_profiles}


The JSON keys must exactly match the provided column names.
Every column must appear exactly once.

Return exactly one valid JSON object.
Do not include ```json.
Do not write prose outside the JSON.
Do not include markdown.

JSON format:

{{
    "COLUMN_NAME": "SEMANTIC_TYPE,
}}
"""
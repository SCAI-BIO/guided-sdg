from typing import Any

def semantic_type_message(column_profiles:dict[str, dict[str, Any]]) -> str:
    return f"""
Current step: Infer semantic types for each column in the dataframe.

Infer the semantic types of each column in the dataframe based on .

Allowed semantic types:
- Float
- Integer
- Categorical
- Ordinal

Column profiles:
{column_profiles}


The JSON keys must exactly match the provided column names.
Every column must appear exactly once.

Return exactly one valid JSON object.
Do not include ```json.

JSON format:

{{
    "COLUMN_NAME": "SEMANTIC_TYPE,
}}
"""
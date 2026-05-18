import pandas as pd
import json
from pathlib import Path

# Store information about the prepared dataset 
# in JSON format for tool use (using the defined 
# semantic types)

def format_json(csv_path:str, semantic_types:dict):
    df = pd.read_csv(csv_path)
    json_path = f"../data/{Path(csv_path).stem}.json" # May want to change the path
    columns = []

    for col in df.columns:
        semantic_type = semantic_types[col]
        if semantic_type in ["Categorical", "Ordinal"]:
            values = sorted(df[col].dropna().astype(str).unique().tolist())

            columns.append({
                "name": col,
                "type": semantic_type,
                "i2s": values,
                "size": len(values)
                })

        elif semantic_type == "Float":
            columns.append({
                "name": col,
                "type": "Float",
                "min": float(df[col].min()),
                "max": float(df[col].max())
                })

        elif semantic_type == "Integer":
            columns.append({
                "name": col,
                "type": "Integer",
                "min": int(df[col].min()),
                "max": int(df[col].max())
                })

    metadata = {"columns": columns}

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return "JSON formatted for tool use."
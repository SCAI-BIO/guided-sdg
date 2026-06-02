import pandas as pd
import json
from pathlib import Path

def format_json(csv_path:str, semantic_types:dict) -> str:
    """Store information about the prepared dataset in JSON file format for use
    of the synthetic generator tool, incorporating the defined semantic types of the data.

    Creates a JSON in data/ with the same name as the csv_path.

    Args:
        csv_path (str): Path of the original dataset. This is used to name 
                        the JSON.
        semantic_types (dict): The defined semantic types of the dataset. This 
                        is used to gather appropriate information in the JSON
                        for each column.

    Returns:
        str: Path to the generated metadata JSON file.
    """
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

        elif semantic_type in ["Integer", "Boolean"]:
            columns.append({
                "name": col,
                "type": "Integer",
                "min": int(df[col].min()),
                "max": int(df[col].max())
                })

    metadata = {"columns": columns}

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"- Created JSON for generator input: {json_path}")
    return json_path
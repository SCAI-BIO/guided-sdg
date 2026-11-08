from pathlib import Path
from typing import Any
import json

import pandas as pd
import requests

from agentic_synth_data.strategies.base import GeneratorStrategy
from agentic_synth_data.utils import DATA_DIR, OUTPUT_DIR, SDR_URL, SemanticTypes


def to_sdr_path(path: Path) -> Path:
    """Convert the local `data/` or `output/` path to its corresponding path in the SDR
    container.

    Args:
        path (Path): Local path to convert.

    Returns:
        Path: Corresponding path inside the SDR container.
    """
    path = path.resolve()

    for local_dir, container_dir in (
        (DATA_DIR, Path("/workspace/data")), (OUTPUT_DIR, Path("/workspace/output")),
    ):
        local_dir = local_dir.resolve()

        if path.is_relative_to(local_dir):
            return container_dir / path.relative_to(local_dir)

    return path


class SyntheticDataReleaseStrategy(GeneratorStrategy):
    """For Synthetic Data Release models: CTGAN, PATEGAN, BayesianNet, and PrivBayes"""
    def prepare_metadata(self,prep_path: Path, semantic_types: SemanticTypes,):
        return self._write_metadata_json(prep_path=prep_path, semantic_types=semantic_types)


    def run(
        self, n_samples: int, prep_path: Path, synth_path: Path, parameters: dict[str, Any],
    ) -> tuple[Path, pd.DataFrame]:
        self._run_generator(
            n_samples=n_samples,
            prep_path=prep_path,
            synth_path=synth_path,
            parameters=parameters,
        )

        # The generators add an "Unnamed: 0" column, so check that all cols match:
        expected_columns = pd.read_csv(prep_path).columns
        synth_df = pd.read_csv(synth_path, index_col=False)

        extra_columns = synth_df.columns.difference(expected_columns)

        if not extra_columns.empty:
            synth_df = synth_df.drop(columns=extra_columns)

        return synth_path, synth_df


    def _write_metadata_json(self, prep_path: Path, semantic_types: SemanticTypes,) -> Path:
        """Store information about the prepared dataset in JSON file format for use
        of the synthetic generator tool, incorporating the defined semantic types of the data.

        Creates a JSON in data/ with the same name as the csv_path.

        Args:
            prep_path (Path): Path of the original dataset. This is used to name the JSON.
            semantic_types (SemanticTypes): The defined semantic types of the dataset. This is
                used to gather appropriate information in the JSON for each column.

        Returns:
            Path: Path to the generated metadata JSON file.
        """
        df = pd.read_csv(prep_path)
        json_path = prep_path.parent / f"{prep_path.stem}.json"
        columns = []

        for col in df.columns:
            semantic_type = semantic_types[col]

            if semantic_type == "Boolean":
                semantic_type = "Categorical"

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
        print(f"- Created JSON for generator input: {json_path}")
        return json_path


    # TODO: If run_generator fails, for example, plug the error message into the agent and try to have the agent repair the issue.
    def _run_generator(
        self, n_samples: int, prep_path: Path, synth_path: Path, parameters: dict[str, Any],
    ) -> Path:
        mechanism = self.generator_name
        parameter_value = ",".join(map(str, parameters.values()))

        print()
        print("-"*25+f"Running {mechanism}"+"-"*25)

        # NOTE: CTGAN does not allow for parameter updates (comma separation issue)
        param_arg = None if mechanism == "CTGAN" else parameter_value

        # The generator CLI expects the datapath without an extension; the
        # co-located <base>.csv and <base>.json are read by the generator.
        payload = {
            "mechanism": mechanism,
            "datapath": str(to_sdr_path(prep_path).with_suffix("")),
            "output_file": str(to_sdr_path(synth_path)),
            "sample_size": n_samples,
            "parameters": param_arg,
        }

        response = requests.post(f"{SDR_URL}/generate", json=payload)

        print(response.json().get("log", ""), end="")

        # If process fails for a specific generator
        if response.status_code != 200:
            raise RuntimeError(
                f"{mechanism} failed. {response.json().get('log', response.text)}"
            )

        print(f"\n🠊 {mechanism}: {n_samples} synthetic samples successfully generated.")
        return synth_path

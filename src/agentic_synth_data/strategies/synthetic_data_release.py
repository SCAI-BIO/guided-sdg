from pathlib import Path
import subprocess as sp
from textwrap import dedent
import json

import pandas as pd

from agentic_synth_data.strategies.base import GeneratorStrategy


class SyntheticDataReleaseStrategy(GeneratorStrategy):
    """For Synthetic Data Release models: CTGAN, PATEGAN, BayesianNet, and PrivBayes"""
    def prepare_metadata(self, prep_path:Path, semantic_types:dict[str,str]):
        return self._write_metadata_json(prep_path=prep_path, semantic_types=semantic_types)


    def run(
            self, n_samples:int, prep_path:Path, synth_path:Path, #parameters:str,
        ) -> tuple[Path, pd.DataFrame]:
        self._run_generator(
            n_samples=n_samples,
            prep_path=prep_path,
            synth_path=synth_path,
            #parameters=parameters
        )

        # The generators add an "Unnamed: 0" column, so check that all cols match:
        expected_columns = pd.read_csv(prep_path).columns
        synth_df = pd.read_csv(synth_path, index_col=False)

        extra_columns = synth_df.columns.difference(expected_columns)

        if not extra_columns.empty:
            synth_df = synth_df.drop(columns=extra_columns)

        return synth_path, synth_df


    def _write_metadata_json(self, prep_path:Path, semantic_types:dict) -> Path:
        """Store information about the prepared dataset in JSON file format for use
        of the synthetic generator tool, incorporating the defined semantic types of the data.

        Creates a JSON in data/ with the same name as the csv_path.

        Args:
            prep_path (Path): Path of the original dataset. This is used to name 
                            the JSON.
            semantic_types (dict): The defined semantic types of the dataset. This 
                            is used to gather appropriate information in the JSON
                            for each column.

        Returns:
            str: Path to the generated metadata JSON file.
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
            self, n_samples:int, prep_path:Path, synth_path:Path, #parameters:str # NOTE: params are not yet merged to dev
        ) -> Path:
        mechanism = self.generator_name
        if mechanism == "CTGAN_sdr":
            mechanism = mechanism.split("_")[0]
        container_path = f"/workspace/output/{synth_path.name}"

        print()
        print("-"*25+f"Running {mechanism}"+"-"*25)

        # NOTE: parameters are not implemented yet here.
        param_arg = "" #if mechanism == "CTGAN" else f"-P {parameters}" # NOTE: CTGAN does not allow for parameter updates (comma separation issue)

        command = dedent(f"""
        podman run --rm \
            --entrypoint /bin/bash -v "$PWD/..:/workspace:Z" \
            docker.io/springepfl/synthetic-data:latest \
            -c "cd /synthetic_data_release && python3 executables/generate_synthetic_dataset.py \
            -D /workspace/data/{prep_path.stem} \
            -M {mechanism} \
            {param_arg} \
            -O {container_path} \
            -N {n_samples}"
            """)

        process = sp.Popen(
            command,
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True
            )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        # If process fails for a specific generator
        if process.returncode !=0:
            raise RuntimeError(f"{mechanism} failed. {command}")
        print(f"\n=> {mechanism}: {n_samples} synthetic samples successfully generated.")
        return synth_path
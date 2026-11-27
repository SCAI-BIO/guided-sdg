from pathlib import Path
import subprocess as sp
from textwrap import dedent

import pandas as pd
import yaml

from agentic_synth_data.generation.strategies.base import GeneratorStrategy
from agentic_synth_data.utils import PROJECT_ROOT # NOTE: might need to update


class Real4RegDatasynth(GeneratorStrategy):
    """For Real4Reg Datasynth models: CTGAN, TabSyn, and STaSY."""
    def prepare_metadata(self, prep_path: Path, stypes: dict[str, str],) -> Path:
        return self._write_metadata_yaml(
            prep_path=prep_path,
            stypes=stypes,
        )


    def run(
        self,
        n_samples: int,
        prep_path: Path,
        synth_path: Path,
    ) -> tuple[Path, pd.DataFrame]:
        self._run_generator(
            n_samples=n_samples,
            prep_path=prep_path,
            synth_path=synth_path,
        )

        synth_df = pd.read_parquet(synth_path)

        return synth_path, synth_df


    def _write_metadata_yaml(
        self,
        prep_path: Path,
        stypes: dict[str, str],
    ) -> Path:
        type_mapping = {
            "Boolean": "categorical",
            "Categorical": "categorical",
            "Ordinal": "categorical", # keep string valued ordinal columns categorical for now.
            "Float": "continuous",
            "Integer": "continuous",
        }

        metadata = {
            "categorical": [],
            "ordinal": [],
            "continuous": [],
        }

        for column, semantic_type in stypes.items():
            datasynth_type = type_mapping[semantic_type]
            metadata[datasynth_type].append(column)

        yaml_path = prep_path.parent / f"{prep_path.stem}_types.yaml"

        with yaml_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(metadata, file, sort_keys=False)

        print(f"- Created YAML for generator input: {yaml_path}")

        return yaml_path


    def _run_generator(
        self,
        n_samples: int,
        prep_path: Path,
        synth_path: Path,
    ) -> Path:
        metadata_path = prep_path.parent / f"{prep_path.stem}_types.yaml"
        mechanism = self.generator_name.lower()
        if mechanism == "ctgan_r4r":
            mechanism = mechanism.split("_")[0]

        synth_path.parent.mkdir(parents=True, exist_ok=True)

        print()
        print("-" * 25 + f"Running {self.generator_name}" + "-" * 25)
        command = dedent(f"""
            "{PROJECT_ROOT / ".generator-envs" / "datasynth" / "bin" / "python"}" -c "
            from datasynth.core.data_model import DataDesign
            from datasynth.generators.tabular.ctgan import CTGANGenerator
            from datasynth.generators.tabular.stasy import STaSyGenerator
            from datasynth.generators.tabular.tabsyn import TabSynGenerator

            prep_path = Path(r'{prep_path.resolve()}')
            metadata_path = Path(r'{metadata_path.resolve()}')
            synth_path = Path(r'{synth_path.resolve()}')
            n_samples = {n_samples}
            mechanism = {mechanism!r}

            if prep_path.suffix.lower() in {{'.parquet', '.pq'}}:
                prepared_df = pd.read_parquet(prep_path)
            else:
                prepared_df = pd.read_csv(prep_path)

            with metadata_path.open('r') as file:
                metadata = yaml.safe_load(file)

            data_design = DataDesign(
                columns=prepared_df.columns.tolist(),
                discrete_columns=metadata.get('categorical', []),
                continuous_columns=metadata.get('continuous', []),
                ordinal_columns=metadata.get('ordinal', []),
            )

            if mechanism == 'ctgan':
                generator = CTGANGenerator()
            if mechanism == 'tabsyn':
                generator = TabSynGenerator()
            if mechanism == 'stasy':
                generator = STaSyGenerator(beta_start=0.0001, beta_end=0.02)


            generator.fit(
                data=prepared_df,
                data_design=data_design,
            )

            synthetic_df = generator.generate(n_samples)

            synth_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            synthetic_df.to_parquet(
                synth_path,
                index=False,
            )

            print(
                f'Generated {{len(synthetic_df)}} rows '
                f'at {{synth_path}}'
            )
            "
        """)

        process = sp.Popen(
            command,
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode !=0:
            raise RuntimeError(f"{mechanism} failed. {command}")

        print(f"\n=> {self.generator_name}: {n_samples} synthetic samples successfully generated.")
        return synth_path

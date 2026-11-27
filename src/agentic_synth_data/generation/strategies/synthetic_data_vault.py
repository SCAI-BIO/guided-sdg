from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

from agentic_synth_data.generation.strategies.base import GeneratorStrategy


class SyntheticDataVaultStrategy(GeneratorStrategy):
    def prepare_metadata(self, prep_path: Path, stypes: dict[str, str],) -> Path:
        df = pd.read_csv(prep_path)
        metadata = Metadata.detect_from_dataframe(
            data=df,
            table_name=prep_path.stem,
            infer_keys=None,
        )

        stype_mapping = {
            "Float": "numerical",
            "Integer": "numerical",
            "Boolean": "boolean",
            "Categorical": "categorical",
            "Ordinal": "categorical",
        }

        for col, stype in stypes.items():
            sdtype = stype_mapping[stype]

            if sdtype == "boolean" and not pd.api.types.is_bool_dtype(df[col]):
                sdtype = "categorical"

            metadata.update_column(
                column_name=col,
                sdtype=sdtype,
            )

        metadata.validate()

        metadata_path = prep_path.parent / f"{prep_path.stem}_metadata.json"

        with TemporaryDirectory(dir=metadata_path.parent) as temp_dir:
            temp_path = Path(temp_dir) / metadata_path.name

            metadata.save_to_json(filepath=temp_path)
            temp_path.replace(metadata_path)

        return metadata_path


    def run(
        self, n_samples: int, prep_path: Path, synth_path: Path, parameters: str,
    ) -> tuple[Path, pd.DataFrame]:
        df = pd.read_csv(prep_path)
        metadata_path = prep_path.parent / f"{prep_path.stem}_metadata.json"
        metadata = Metadata.load_from_json(filepath=metadata_path)
        mechanism = self.generator_name

        if mechanism == "CTGAN_sdv":
            synthesizer = CTGANSynthesizer(metadata=metadata,)
            # TODO: think about how to include parameters

        elif mechanism == "TVAE":
            synthesizer = TVAESynthesizer(metadata=metadata,)

        else: # "GaussianCopula"
            synthesizer = GaussianCopulaSynthesizer(metadata=metadata)

        synthesizer.fit(df)
        synth_df = synthesizer.sample(num_rows=n_samples)

        synth_path.parent.mkdir(parents=True, exist_ok=True,)
        synth_df.to_csv(synth_path, index=False,)

        return synth_path, synth_df

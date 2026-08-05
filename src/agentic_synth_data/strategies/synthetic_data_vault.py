from pathlib import Path

import pandas as pd

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

from agentic_synth_data.strategies.base import GeneratorStrategy


class SyntheticDataVaultStrategy(GeneratorStrategy):
    def prepare_metadata(self, prep_path:Path, semantic_types:dict[str,str]):
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

        for col, stype in semantic_types.items():
            metadata.update_column(
                column_name=col,
                sdtype=stype_mapping[stype],
            )

        metadata.validate()

        metadata_path = prep_path.parent / f"{prep_path.stem}_metadata.json"

        metadata.save_to_json(filepath=metadata_path, mode="overwrite",)
        return metadata


    def run(
            self, n_samples:int, prep_path:Path, synth_path:Path, #parameters:str,
        ) -> tuple[Path, pd.DataFrame]:
        df = pd.read_csv(prep_path)
        metadata_path = prep_path.parent / f"{prep_path.stem}_metadata.json"
        metadata = Metadata.load_from_json(filepath=metadata_path)
        mechanism = self.generator_name

        if mechanism == "CTGAN_sdv":
            synthesizer = CTGANSynthesizer(metadata=metadata,) 
            # think about how to include parameters

        elif mechanism == "TVAE":
            synthesizer = TVAESynthesizer(metadata=metadata,)

        elif mechanism == "GaussianCopula":
            synthesizer = GaussianCopulaSynthesizer(metadata=metadata)

        synthesizer.fit(df)
        synth_df = synthesizer.sample(num_rows=n_samples)

        synth_path.parent.mkdir(parents=True, exist_ok=True,)
        synth_df.to_csv(synth_path, index=False,)
        return synth_path, synth_df

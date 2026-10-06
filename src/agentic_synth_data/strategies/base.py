from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import pandas as pd


class GeneratorStrategy(ABC):
    def __init__(self, generator_name: str,):
        self.generator_name = generator_name


    def _make_synth_path(self, prep_path: Path, version_id: int,) -> Path:
        """Format the path for the synthetic data.

        Args:
            prep_path (Path): Prepared dataset path.
            version_id (int): Current DataGen version ID.

        Returns:
            Path: Synthetic dataset path.
        """
        data_name = prep_path.stem.removesuffix("_prepared")

        output_dir = Path("output") / f"version_{version_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        synth_path = f"synthetic_{data_name}_{self.generator_name.lower()}_v{version_id}.csv"
        return output_dir / synth_path


    def _run(self, n_samples: int, prep_path: Path, version_id: int, parameters: dict[str, Any],):
        synth_path = self._make_synth_path(prep_path=prep_path, version_id=version_id)
        return self.run(
            n_samples=n_samples,
            prep_path=prep_path,
            synth_path=synth_path,
            parameters=parameters,
        )


    @abstractmethod
    def prepare_metadata(
        self,
        prep_path: Path,
        semantic_types: dict[str, Literal["Float", "Integer", "Boolean", "Categorical", "Ordinal"]],
    ) -> Path:
        """Prepare metadata for tool use.

        Args:
            prep_path (Path): Prepared dataset path.
            semantic_types (dict[str,str]): Semantic types assigned to each column.

        Returns:
            Path: Metadata path.
        """


    @abstractmethod
    def run(
        self, n_samples: int, prep_path: Path, synth_path: Path, parameters: dict[str, Any],
    ) -> tuple[Path, pd.DataFrame]:
        """Run selected mechanism.

        Args:
            n_samples (int): Number of samples to generate.
            prep_path (Path): Prepared dataset path.
            synth_path (Path): Synthetic dataset path

        Returns:
            tuple[Path, pd.DataFrame]: Synthetic path and dataframe.
        """

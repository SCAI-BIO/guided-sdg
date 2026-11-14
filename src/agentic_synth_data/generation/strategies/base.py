from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from agentic_synth_data.utils import OUTPUT_DIR, SemanticTypes


class GeneratorStrategy(ABC):
    """Abstract base class for synthetic data generation strategies. Defines a common interface
    for preparing metadata and generating synthetic data using different generator implementations.
    """
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

        output_dir = OUTPUT_DIR / f"version_{version_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        synth_path = f"synthetic_{data_name}_{self.generator_name.lower()}_v{version_id}.csv"
        return output_dir / synth_path


    def _run(self, n_samples: int, prep_path: Path, version_id: int, parameters: dict[str, Any],):
        synth_path = self._make_synth_path(prep_path=prep_path, version_id=version_id,)
        return self.run(
            n_samples=n_samples,
            prep_path=prep_path,
            synth_path=synth_path,
            parameters=parameters,
        )


    @abstractmethod
    def prepare_metadata(self, prep_path: Path, stypes: SemanticTypes,) -> Path:
        """Prepare metadata for tool use.

        Args:
            prep_path (Path): Prepared dataset path.
            stypes (dict[str,str]): Semantic types assigned to each column.

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
            synth_path (Path): Synthetic dataset path.
            parameters (dict[str, Any]): Parameter configuration.

        Returns:
            tuple[Path, pd.DataFrame]: Synthetic path and dataframe.
        """

from agentic_synth_data.generation.strategies.base import GeneratorStrategy
from agentic_synth_data.generation.strategies.synthetic_data_release import SyntheticDataReleaseStrategy
from agentic_synth_data.generation.strategies.r4r_datasynth import Real4RegDatasynth
from agentic_synth_data.generation.strategies.synthetic_data_vault import SyntheticDataVaultStrategy


GENERATOR_BACKENDS = {
    "CTGAN_sdr": "synthetic_data_release",
    "PATEGAN": "synthetic_data_release",
    "PrivBayes": "synthetic_data_release",
    "BayesianNet": "synthetic_data_release",
    "CTGAN_r4r": "r4r_datasynth",
    "STaSY": "r4r_datasynth",
    "TabSyn": "r4r_datasynth",
    "CTGAN_sdv": "synthetic_data_vault",
    "TVAE": "synthetic_data_vault",
    "GaussianCopula": "synthetic_data_vault",
}


def get_generator_strategy(generator_name: str) -> GeneratorStrategy:
    """Return the generator strategy for the selected backend.

    Args:
        generator_name (str): Name of the selected generator.

    Returns:
        GeneratorStrategy: Generator strategy corresponding to the selected generator.
    """
    backend = GENERATOR_BACKENDS[generator_name]

    if backend == "synthetic_data_release":
        return SyntheticDataReleaseStrategy(generator_name)

    if backend == "r4r_datasynth":
        return Real4RegDatasynth(generator_name)

    if backend == "synthetic_data_vault":
        return SyntheticDataVaultStrategy(generator_name)

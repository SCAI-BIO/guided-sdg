from agentic_synth_data.generation.strategies.base import GeneratorStrategy
from agentic_synth_data.generation.strategies.synthetic_data_release import SyntheticDataReleaseStrategy


GENERATOR_BACKENDS = {
    "CTGAN": "synthetic_data_release",
    "PATEGAN": "synthetic_data_release",
    "PrivBayes": "synthetic_data_release",
    "BayesianNet": "synthetic_data_release",
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

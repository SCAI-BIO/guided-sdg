from agentic_synth_data.strategies.base import GeneratorStrategy
from agentic_synth_data.strategies.r4r_datasynth import Real4RegDatasynth
from agentic_synth_data.strategies.synthetic_data_release import SyntheticDataReleaseStrategy


GENERATOR_BACKENDS = {
    "CTGAN_sdr": "synthetic_data_release",
    "PATEGAN": "synthetic_data_release",
    "PrivBayes": "synthetic_data_release",
    "BayesianNet": "synthetic_data_release",
    "CTGAN_r4r": "r4r_datasynth",
    "STaSY": "r4r_datasynth",
    "TabSyn": "r4r_datasynth",
}

def get_generator_strategy(generator_name:str) -> GeneratorStrategy:
    backend = GENERATOR_BACKENDS[generator_name]

    if backend ==  "synthetic_data_release":
        return SyntheticDataReleaseStrategy(generator_name)

    if backend == "r4r_datasynth":
        return Real4RegDatasynth(generator_name)

from typing import Any, Literal

from pydantic import Field, BaseModel

from agentic_synth_data.utils import CategoricalChoiceAuto, GeneratorNameAuto, NumericalChoiceAuto


class SampleSizeConfig(BaseModel):
    """Synthesis request sample size configuration."""
    mode: Literal["fixed_value", "same_as_original"] | None = None
    fixed_value: int | None = None


class PrivacyConfig(BaseModel):
    """Synthesis request privacy configuration."""
    mode: Literal["standard", "differential_privacy"] = "standard"
    sensitive_columns: list[str] = Field(default_factory=list)


class GeneratorConfig(BaseModel):
    """Synthesis request generator configuration."""
    name: GeneratorNameAuto | None = None
    parameters: dict[str, Any] | None = None


class ImputationConfig(BaseModel):
    """Synthesis request imputation configuration."""
    method: Literal["SimpleImputer"] | None = None
    categorical_strategy: CategoricalChoiceAuto | None = None
    numerical_strategy: NumericalChoiceAuto | None = None


class MissingValueConfig(BaseModel):
    """Synthesis request missing value handling configuration."""
    mode: Literal["auto", "impute", "remove"] | None = None
    impute: ImputationConfig = Field(default_factory=ImputationConfig)


class SynthesisRequest(BaseModel):
    """Configured synthesis request."""
    sample_size: SampleSizeConfig = Field(default_factory=SampleSizeConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    missing_values: MissingValueConfig = Field(default_factory=MissingValueConfig)

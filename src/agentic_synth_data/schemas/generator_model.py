from typing import Any

from pydantic import BaseModel, Field

from agentic_synth_data.utils import GeneratorName


class SelectGenerator(BaseModel):
    """Structured LLM assessment of a synthetic data generator."""
    generator: GeneratorName
    pros: str
    cons: str
    rationale: str
    warning: list[str]


class GeneratorRevisionOption(BaseModel):
    """Proposed generator configuration for workflow revision."""
    id: str
    generator_name: GeneratorName
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str


class GeneratorRevisionPlan(BaseModel):
    """Contains proposed generator revision options."""
    options: list[GeneratorRevisionOption]


class PATEGANConfig(BaseModel):
    """Configuration parameters for the PATEGAN generator. Epsilon only."""
    eps: int = Field(default=1, ge=0, le=7)


class PrivBayesConfig(BaseModel):
    """Configuration parameters for the PrivBayes generator. Epsilon only."""
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)

from typing import Literal

from pydantic import BaseModel, Field

from agentic_synth_data.utils import CategoricalChoice, NumericalChoice


class MissingRevisionOption(BaseModel):
    """Proposed missing value handling strategy for workflow revision."""
    id: str
    mode: Literal["remove", "impute",]
    imputation_method: Literal["SimpleImputer"] | None = None
    categorical_strategy: CategoricalChoice | None = None
    numerical_strategy: NumericalChoice | None = None
    reason: str


class MissingRevisionPlan(BaseModel):
    """Contains proposed missing value handling revisions."""
    options: list[MissingRevisionOption]


class MissingHandling(BaseModel):
    """Structured LLM recommendation for handling missing values."""
    mode: str
    set_threshold: int | None = None
    imputation_method: str | None = None
    categorical_strategy: str | None = None
    numerical_strategy: str | None = None
    warning: list[str] = Field(default_factory=list)
    rationale: str

from collections.abc import Hashable
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

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
    imputation_method: str | None = None
    categorical_strategy: str | None = None
    numerical_strategy: str | None = None
    warning: list[str] = Field(default_factory=list)
    rationale: str


class MissingPrecheckReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) # Allow pd.DataFrame type
    df: pd.DataFrame
    has_missing_markers: bool
    detected_markers: dict[Hashable, int] = Field(default_factory=dict)
    replaced_markers: list[Hashable] = Field(default_factory=list)

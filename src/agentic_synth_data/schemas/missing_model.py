from typing import Literal

from pydantic import BaseModel, Field


class MissingRevisionOption(BaseModel):
    """Proposed missing value handling strategy for workflow revision."""
    id: str
    mode: Literal["remove", "impute",]
    imputation_method: Literal["SimpleImputer"] | None = None
    categorical_strategy: Literal["most_frequent", "constant",] | None = None
    numerical_strategy: Literal["mean", "median", "most_frequent",] | None = None
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

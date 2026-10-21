from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field

from anonymeter.stats.confidence import PrivacyRisk


@dataclass
class AnonymeterInput:
    """Contains prepared real and synthetic dataframe copies with missing value markers applied for 
    compatibility with Anonymeter functions, which do not accept null values."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    synthetic_df: pd.DataFrame
    missing_markers: dict[str, str | int | float]


class AttackColumns(BaseModel):
    """Agent suggested attack columns for eval."""
    linkability_aux_cols: list[list[str]] = Field(min_length=2, max_length=2)
    inference_aux_cols: list[str] = Field(min_length=1) # possible TODO: Add additional constraints
    secret: str
    rationale: str


class AttackOutputs(BaseModel):
    """Anonymeter warnings."""
    singling_out: str | None = None
    inference: str | None = None
    linkability: str | None = None


class AttackMetrics(BaseModel):
    """Final Anonymeter metrics, agent column choices, and tool outputs."""
    singling_out: PrivacyRisk
    inference: PrivacyRisk
    linkability: PrivacyRisk
    selected_cols: AttackColumns
    output_messages: AttackOutputs

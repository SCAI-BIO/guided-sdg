from pydantic import BaseModel, Field

from anonymeter.stats.confidence import PrivacyRisk


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

from pydantic import BaseModel, Field


class SyndatOutputs(BaseModel):
    """Syndat warnings."""
    jsd: str | None = None
    norm_corr_diff: str | None = None
    discrim_auc: str | None = None


class SyndatScores(BaseModel):
    """Normalized Syndat fidelity scores."""
    distribution: float | None = None
    correlation: float | None = None
    discrimination: float | None = None


class SyndatMetrics(BaseModel):
    """Final Syndat metrics with function outputs (if exists)."""
    jsd: dict | None = None
    norm_corr_diff: float | None = None
    discrim_auc: float | None = None
    normalized_scores: SyndatScores = Field(default_factory=SyndatScores)
    output_messages: SyndatOutputs = Field(default_factory=SyndatOutputs)

from pydantic import BaseModel


class SyndatOutputs(BaseModel):
    """Syndat warnings."""
    jsd: str | None = None
    norm_corr_diff: str | None = None
    discrim_auc: str | None = None


class SyndatMetrics(BaseModel):
    """Final syndat metrics with function outputs (if exists)."""
    jsd: dict | None = None
    norm_corr_diff: float | None = None
    discrim_auc: float | None = None
    output_messages: SyndatOutputs

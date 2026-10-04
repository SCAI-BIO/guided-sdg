from typing import Literal

from pydantic import Field, BaseModel


class AgentEval(BaseModel):
    status: Literal["satisfactory", "needs_review", "not_satisfactory"]
    problem_columns: list[str] = Field(default_factory=list)
    recommendation: str
    rationale: str


class SyndatOutputs(BaseModel):
    """Syndat function outputs for each metric"""
    jsd: str | None = None
    norm_corr_diff: str | None = None
    discrim_auc: str | None = None

from typing import Literal

from pydantic import Field, BaseModel


class AgentEval(BaseModel):
    """Structured LLM assessment of synthetic data evaluation results."""
    status: Literal["satisfactory", "needs_review", "not_satisfactory"]
    problem_columns: list[str] = Field(default_factory=list)
    recommendation: str
    rationale: str

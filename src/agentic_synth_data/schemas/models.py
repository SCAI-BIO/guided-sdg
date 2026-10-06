from typing import Literal

from pydantic import Field, BaseModel


class AgentEval(BaseModel):
    status: Literal["satisfactory", "needs_review", "not_satisfactory"]
    problem_columns: list[str] = Field(default_factory=list)
    recommendation: str
    rationale: str

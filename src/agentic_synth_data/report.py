from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic_synth_data.schemas.attack_model import AttackMetrics
from agentic_synth_data.schemas.syndat_model import SyndatMetrics


class InputReport(BaseModel):
    """Initial dataset review and version for report."""
    path: str | None = None
    original_shape: tuple[int, int] | None = None
    prepared_shape: tuple[int, int] | None = None
    version_id: int | None = None
    parent_version: int | None = None


class PrepReport(BaseModel):
    """Preparation steps and information gathered prior to generation for report."""
    missing_markers_detected: dict = Field(default_factory=dict)
    missing_markers_replaced: dict = Field(default_factory=dict)
    dtypes: dict = Field(default_factory=dict)
    stypes: dict = Field(default_factory=dict)
    profile: dict = Field(default_factory=dict)
    missing_before: dict = Field(default_factory=dict)
    missing_after: dict = Field(default_factory=dict)


class GenerationReport(BaseModel):
    """Information gathered from synthetic data generation for report."""
    prep_path: str | None = None
    synth_path: str | None = None
    synth_shape: tuple[int, int] | None = None
    params: dict[str, Any] | None = None


class EvalReport(BaseModel):
    """Evaluation metrics with plots comparing real and synthetic data for report."""
    attack: AttackMetrics | None = None
    syndat: SyndatMetrics | None = None
    plots: dict[str, list[Path]] = Field(default_factory=dict)


class ReviewReport(BaseModel):
    # TODO: After splitting, as this is, it only has the same information as AgentEval. Possibly add more regarding user input in response.
    """Agent identified input for report."""
    problem_columns: list[str] = Field(default_factory=list)
    status: str | None = None
    recommendation: str | None = None
    rationale: str | None = None
    # user_decision - finish | revise
    # additional details...


class UpdatePlanReport(BaseModel):
    """The plan update that was selected for a subsequent run to be included in the current report.
    """
    selected_plan_updates: dict = Field(default_factory=dict)


class Report(BaseModel):
    """Report of main changes made during a given workflow iteration."""
    input: InputReport = Field(default_factory=InputReport)
    prep: PrepReport = Field(default_factory=PrepReport)
    gen: GenerationReport = Field(default_factory=GenerationReport)
    evaluate: EvalReport = Field(default_factory=EvalReport)
    review: ReviewReport = Field(default_factory=ReviewReport)
    update: UpdatePlanReport = Field(default_factory=UpdatePlanReport)
    revisions: list = Field(default_factory=list)

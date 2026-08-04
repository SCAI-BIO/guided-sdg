from pydantic import BaseModel, Field


class InputReport(BaseModel):
    path: str | None = None
    original_shape: tuple[int, int] | None = None
    prepared_shape: tuple[int, int] | None = None
    version_id: int | None = None
    parent_version: int | None = None


class PrepReport(BaseModel):
    missing_markers_detected: dict = Field(default_factory=dict)
    missing_markers_replaced: dict = Field(default_factory=dict)
    dtypes: dict = Field(default_factory=dict)
    stypes: dict = Field(default_factory=dict)
    profile: dict = Field(default_factory=dict)
    missing_before: dict = Field(default_factory=dict)
    missing_after: dict = Field(default_factory=dict)


class GenerationReport(BaseModel):
    prep_path: str | None = None
    synth_path: str | None = None
    synth_shape: tuple[int, int] | None = None
    params: str | None = None


class ReviewReport(BaseModel):
    jsd: dict = Field(default_factory=dict)
    norm_corr_diff: float | None = None
    discrim_auc: float | None = None
    problem_columns: list[str] = Field(default_factory=list)
    status: str | None = None
    recommendation: str | None = None
    rationale: str | None = None
    selected_plan_updates: dict | None = None


class Report(BaseModel):
    input: InputReport = Field(default_factory=InputReport)
    prep: PrepReport = Field(default_factory=PrepReport)
    gen: GenerationReport = Field(default_factory=GenerationReport)
    review: ReviewReport = Field(default_factory=ReviewReport)
    revisions: list = Field(default_factory=list)

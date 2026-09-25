from pydantic import BaseModel, Field

from anonymeter.stats.confidence import PrivacyRisk

class AttackColumns(BaseModel):
    linkability_aux_cols:list[list[str]] = Field(min_length=2, max_length=2)
    inference_aux_cols:list[str] = Field(min_length=1) # possible TODO: Add additional constraints
    secret:str
    rationale:str


class AttackMetrics(BaseModel):
    singling_out: PrivacyRisk
    inference: PrivacyRisk
    linkability: PrivacyRisk
    selected_cols: AttackColumns

from pydantic import BaseModel

from anonymeter.stats.confidence import PrivacyRisk


class AttackMetrics(BaseModel):
    singling_out: PrivacyRisk
    inference: PrivacyRisk
    linkability: PrivacyRisk
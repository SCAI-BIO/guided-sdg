from typing import Any

from agentic_synth_data.preference_setup import SynthesisRequest


def revision_message(
        synthesis_request:SynthesisRequest, agent_decision:dict[str, Any],
    ) -> str:
    return f"""
The workflow should be revised.

Current synthesis request:
{synthesis_request.model_dump()}

Evaluation decision:
{agent_decision}

Allowed revision scope:
- You may only suggest changes to generator selection.
- Do not suggest missing value handling changes.
- Do not suggest dtype changes.
- Do not suggest semantic type changes.
- Do not suggest sample size changes.
- Do not suggest privacy mode changes.

Available generators:
- CTGAN
- PATEGAN
- BayesianNet
- PrivBayes

Return valid JSON only with this structure:

{{
  "revision_options": [
    {{
      "id": "option_1",
      "rerun_step": "generator_selection",
      "generator_name": "CTGAN",
      "parameters": {{}},
      "reason": "short reason based on evaluation metrics"
    }}
  ]
}}

Give 1 to 3 options.
"""
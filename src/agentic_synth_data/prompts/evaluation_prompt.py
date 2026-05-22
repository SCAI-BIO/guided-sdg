from typing import Any

def evaluation_prompt(
        synthesis_request:dict[str, Any], 
        generator_used:str, # Already gathered from synthesis request, can be removed.
        jsd_results:dict
    ) -> str:
    return f"""
Evaluate whether synthetic data quality is satisfactory.

Return exactly ONE valid JSON object.
No markdown.
No explanation outside of JSON.

Synthesis request:
{synthesis_request}

Generator used:
{generator_used}

Jensen shannon distance results:
{jsd_results}

Evaluate whether the synthetic data is satisfactory.

Interpretation:
- Jensen-Shannon distance is between 0 and 1.
- 0 means the distributions are identical.
- average_jsd <= 0.10: satisfactory
- 0.10 < average_jsd <= 0.25: acceptable but needs review
- average_jsd > 0.25: not satisfactory
- Columns with very high JSD (>0.50) are problem columns
- whether the selected generator fits the data type

JSON format:
{{
"evaluation": {{
    "status": "satisfactory",
    "average_jsd": 0.0,
    "problem_columns": [],
    "recommendation": "Specific next action. Mention which workflow step should be revisited if needed.",
    "rationale": "Explain why this status was assigned, referring to average JSD, problem columns, generator fit, and any relevant workflow step."
}}
}}
"""

# TODO: Add more instruction behind the evaluation.
# possibly add whether reviisting steps is recommended
# And specifying where to revisit
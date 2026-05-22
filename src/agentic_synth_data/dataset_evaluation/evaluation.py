import json
import pprint
from typing import Optional, Any
import pandas as pd

from syndat.metrics import jensen_shannon_distance, normalized_correlation_difference, discriminator_auc

from langgraph.graph.state import CompiledStateGraph

from agentic_synth_data.utils import format_response
from agentic_synth_data.prompts.evaluation_prompt import evaluation_prompt


# possible TODO: hardcoding a threshold to limit the chance that the agent hallucinates information.
def evaluation(
        real_df:pd.DataFrame, 
        synthetic_df:pd.DataFrame, 
        synthesis_request:dict[str, Any],
        agent:Optional[CompiledStateGraph] | None = None, 
        config:dict | None = None
        ):
    print("\n\n")
    print("="*25+"[STEP: Evaluate synthesized data]"+"="*25)
    generator_used = synthesis_request["generator"]
    jsd_results = jensen_shannon_distance(real=real_df, synthetic=synthetic_df)
    print("Jenson-Shannon Distances:")
    pprint.pprint(jsd_results)

    prompt = evaluation_prompt(
        synthesis_request=synthesis_request, 
        generator_used=generator_used,
        jsd_results=jsd_results,
        )
    
    plan = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        },
        config=config,
    )

    plan = plan["messages"][-1].content.split("</think>")[1].strip()
    parsed = json.loads(plan)
    evaluation = parsed["evaluation"]

    print("\nEvaluation:")
    print(f"- Status: {evaluation['status']}")
    print(f"- Average Jenson-Shannon distance: {evaluation['average_jsd']}")
    print(f"- Problem columns: {evaluation['problem_columns']}")
    print(f"- Recommendation:")
    format_response(evaluation["recommendation"])
    print("- Rationale:")
    format_response(evaluation["rationale"])


# TODO: Add a plan for improvement of the workflow
# the user shoudl be able to confirm their preferences 
# and start a new attempt trying something different at 
# the exact stage desired.

# TODO make a tool for the agent to rerun synthetic data generation method
# can be expanded to additional steps later.
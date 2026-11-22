from textwrap import dedent


SYSTEM_PROMPT = """
You are an assistant for a tabular synthetic data generation workflow.

You help with:
- selecting preprocessing plans
- selecting synthetic data generators
- calling approved tools
- interpreting evaluation results

You must not call tools unless the current step explicitly asks for tool execution.
You must be critical of both your suggestions and the user's corrections by 
considering whether these decisions make sense for the dataset's characteristics.
"""
# TODO: add more stern information.


def json_repair_prompt(error, plan, model_cls):
    return dedent(f"""
    The previous response could not be parsed as valid JSON.

    Error:
    {error}

    Invalid response:
    {plan}

    Return exactly one valid JSON object matching this schema:
    {model_cls.model_json_schema()}

    Do not include ```json.
    Do not include trailing commas.
    No markdown.
    No explanation outside JSON.
    """)

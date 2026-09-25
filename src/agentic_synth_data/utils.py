import textwrap
import streamlit as st
from typing import Literal, Optional, Type, TypeVar

import pandas as pd
from pydantic import BaseModel
from langgraph.graph.state import CompiledStateGraph

from agentic_synth_data.prompts.system_prompt import json_repair_prompt


T = TypeVar("T", bound=BaseModel)

def format_response(text:str):
    print(textwrap.fill(text, width=100))


def display_synthesis_request(synthesis_request:"SynthesisRequest") -> None:
    print(f"\nSample size mode: {synthesis_request.sample_size.mode}")

    if synthesis_request.sample_size.fixed_value is not None:
        print(f"Sample size: {synthesis_request.sample_size.fixed_value}")

    print(f"Privacy mode: {synthesis_request.privacy.mode}")

    print(f"Generator: {synthesis_request.generator.name}")

    print(f"Missing value handling: {synthesis_request.missing_values.mode}")

    if synthesis_request.missing_values.mode == "impute":
        print(f"- Imputer: {synthesis_request.missing_values.impute.method}")
        print(
            "- Categorical strategy: " \
            f"{synthesis_request.missing_values.impute.categorical_strategy}"
        )
        print(f"- Numerical strategy: {synthesis_request.missing_values.impute.numerical_strategy}")


def display_synthesis_request_st(synthesis_request:"SynthesisRequest") -> None:
    st.write(f"\n**Sample size mode:** {synthesis_request.sample_size.mode}")

    if synthesis_request.sample_size.fixed_value is not None:
        st.write(f"**Sample size:** {synthesis_request.sample_size.fixed_value}")

    st.write(f"**Privacy mode:** {synthesis_request.privacy.mode}")

    st.write(f"**Generator:** {synthesis_request.generator.name}")

    st.write(f"**Missing value handling:** {synthesis_request.missing_values.mode}")

    if synthesis_request.missing_values.mode == "impute":
        st.write(f"- **Imputer:** {synthesis_request.missing_values.impute.method}")
        st.write(
            "- **Categorical strategy:** " \
            f"{synthesis_request.missing_values.impute.categorical_strategy}"
        )
        st.write(
            "- **Numerical strategy:** " \
            f"{synthesis_request.missing_values.impute.numerical_strategy}"
        )


def df_formatter_st(df:pd.DataFrame) -> None:
    """Format dataframes with consistent sizing when over 10 rows are present.

    Args:
        df (pd.DataFrame): The input dataframe.
    """
    if len(df) > 10:
        st.dataframe(df, hide_index=True, width="stretch", height=400,)
    else:
        st.dataframe(df, hide_index=True, width="stretch",)


def info_helper_message(message:str, context:str, key:str):
    """Display an information message with a hover help icon.

    Args:
        message (str): Content of the information message.
        context (str): Additional information (e.g., what context was given).
        key (str): Unique key for the widget.
    """
    col1, col2 = st.columns([.95, .05])

    with col1:
        st.info(message)

    with col2:
        st.button(
            "",
            icon=":material/help:",
            help=context,
            type="tertiary",
            key=key,
        )


def parse_json(
        plan:str,
        model_cls:Type[BaseModel],
        agent:Optional[CompiledStateGraph],
        config:dict,
        root:bool=False
    ) -> T:
    """If agent fails to correctly format JSON response, prompts to 
    try again.

    Args:
        plan (str): The agent output.
        model_cls (Type[BaseModel]): Pydantic model class used to validate JSON response.
        agent (Optional[CompiledStateGraph]): LangGraph agent.
        config (dict): Configuration passed to the agent during invocation.
        root (bool, optional): Indicates whether in semantic_type_handling (True) or not. 
                    Defaults to False.

    Returns:
        T: Validated instance of the given Pydantic model.
    """
    if root: # semantic_type_handling follows a different structure than the other steps
        try:
            return model_cls.model_validate_json(plan).root

        except Exception as error:
            print("\nAgent response could not be parsed. Asking agent to repair the response...")

            repair_prompt = json_repair_prompt(error, plan, model_cls)

            repair_raw = agent.invoke(
                {"messages": [{"role": "system", "content": repair_prompt,}]},
                config=config,
            )

            repaired_plan = extract_output(repair_raw)

            return model_cls.model_validate_json(repaired_plan).root
    else:
        try:
            return model_cls.model_validate_json(plan)

        except Exception as error:
            print("\nAgent response could not be parsed. Asking agent to repair the response...")

            repair_prompt = json_repair_prompt(error, plan, model_cls)

            repair_raw = agent.invoke(
                {"messages": [{"role": "system", "content": repair_prompt,}]},
                config=config,
            )

            repaired_plan = extract_output(repair_raw)

            return model_cls.model_validate_json(repaired_plan)


def parse_json_ui(
    plan:str,
    model_cls:Type[BaseModel],
    agent:Optional[CompiledStateGraph],
    config:dict,
    root:bool=False,
) -> T:
    """If agent fails to correctly format JSON response, prompts to 
    try again (UI).

    Args:
        plan (str): The agent output.
        model_cls (Type[BaseModel]): Pydantic model class used to validate JSON response.
        agent (Optional[CompiledStateGraph]): LangGraph agent.
        config (dict): Configuration passed to the agent during invocation.
        root (bool, optional): Indicates whether in semantic_type_handling (True) or not. 
                    Defaults to False.

    Returns:
        T: Validated instance of the given Pydantic model.
    """
    if root: # semantic_type_handling follows a different structure than the other steps
        try:
            return model_cls.model_validate_json(plan).root

        except Exception as error:
            st.warning("Agent response could not be parsed. Asking agent to repair the response...")

            repair_prompt = json_repair_prompt(error, plan, model_cls)

            repair_raw = agent.invoke(
                {"messages": [{"role": "system", "content": repair_prompt,}]},
                config=config,
            )

            repaired_plan = extract_output(repair_raw)

            return model_cls.model_validate_json(repaired_plan).root
    else:
        try:
            return model_cls.model_validate_json(plan)

        except Exception as error:
            st.warning("Agent response could not be parsed. Asking agent to repair the response...")

            repair_prompt = json_repair_prompt(error, plan, model_cls)

            repair_raw = agent.invoke(
                {"messages": [{"role": "system", "content": repair_prompt,}]},
                config=config,
            )

            repaired_plan = extract_output(repair_raw)

            return model_cls.model_validate_json(repaired_plan)


def extract_output(raw_message:dict) -> str:
    """Extract content from the agent's invocation output

    Args:
        raw_message (dict): Raw agent invocation output.

    Returns:
        str: Agent message.
    """
    response = raw_message["messages"][-1].content

    if "</think>" in response:
        return response.split("</think>")[1].strip()

    return response.strip()


def newline_every_n_words(option:str, n:int) -> str:
    """Format the options output by the agent to improve the readability of pyinputplus's 
    choice menu.

    Args:
        option (str): The option output by the agent.
        n (int): The number of words allowed before a newline is added.

    Returns:
        str: The option output by agent with a newline every n words.
    """
    words = option.split()
    lines = []

    for i in range(0, len(words), n):
        lines.append(" ".join(words[i:i+n]))

    return "\n".join(lines)


def output_default_parameters(
        generator_name:Literal["PATEGAN", "BayesianNet", "PrivBayes", "CTGAN", "auto"]
    ) -> dict | None:
    """Takes in generator name and outputs its default parameters. Used in select preferences
    and other instances where parameters are set to default.

    Args:
        generator_name (Literal["PATEGAN", "BayesianNet", "PrivBayes", "CTGAN", "auto"]): Selected.

    Returns:
        str | None: 
        - str: The colon-separated generator parameters.
        - None: If auto is selected, no parameters are returned.
    """
    # Synthetic Data Release
    if generator_name == "PATEGAN":
        return {
            "eps":1, "delta":1e-5, "infer_ranges":False, "num_teachers":10, "n_iters":100,
            "batch_size":128, "learning_rate":1e-4, "multiprocess":False,
        }

    if generator_name == "BayesianNet":
        return {
            "histogram_bins":10, "degree":1, "infer_ranges":False, "multiprocess":True, "seed":None,
        }

    if generator_name == "PrivBayes":
        return {
            "histogram_bins":10, "degree":1, "epsilon":.1, "infer_ranges":False,
            "multiprocess":True, "seed":None,
        }

    if generator_name == "CTGAN":
        return {
            "embedding_dim":128, "gen_dim":(256, 256), "dis_dim":(256, 256), "l2scale":1e-6,
            "batch_size":500, "epochs":300, "multiprocess":False,
        }

    return None # "auto"

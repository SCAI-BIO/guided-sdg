import os
import textwrap
from typing import Any, Literal, Optional, TypeVar, overload
from pathlib import Path

from langchain_core.runnables import RunnableConfig
import pandas as pd
from pandas.io.formats.style import Styler
from pydantic import BaseModel
from langgraph.graph.state import CompiledStateGraph
import streamlit as st
import st_yled

from agentic_synth_data.prompts.system_prompt import json_repair_prompt


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DATA_DIR = PROJECT_ROOT / "demo_data"

# Shared data/output directories. Overridable so that the app and the
# synthetic-data-release sidecar agree on a single path prefix (see compose.yaml).
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output"))

# Base URL of the synthetic-data-release sidecar service.
SDR_URL = os.environ.get("SDR_URL", "http://localhost:8000")


# ============================================================================
# Type aliases
# ============================================================================

T = TypeVar("T", bound=BaseModel)

# Permitted semantic types
type SemanticType = Literal["Float", "Integer", "Boolean", "Categorical", "Ordinal",]

type SemanticTypes = dict[str, SemanticType]

# Generator specification
# a: Primary options
type GeneratorName = Literal["PATEGAN", "CTGAN", "PrivBayes", "BayesianNet",]

# b: If agent-assisted selection is permitted.
type GeneratorNameAuto = Literal["auto", "PATEGAN", "CTGAN", "PrivBayes", "BayesianNet",]

# Imputation specification for categorical or numerical null values
# a: Primary options
type CategoricalChoice = Literal["most_frequent", "constant",]

type NumericalChoice = Literal["mean", "median", "most_frequent",]

# b: If agent-assisted selection is permitted
type CategoricalChoiceAuto = Literal["auto", "most_frequent", "constant",]

type NumericalChoiceAuto = Literal["auto", "mean", "median", "most_frequent",]


# ============================================================================
# Basic text display helpers
# ============================================================================

def comma_sep(item_list: list) -> str:
    """Join list items into a comma separated string.

    Args:
        item_list (list): list of items.
    """
    return ", ".join(item_list)


def format_option(value: str) -> str:
    """Replace underscores with spaces for display (UI).

    Args:
        value (str): Internal string with underscores.

    Returns:
        str: String with spaces instead of underscores.
    """
    return value.replace("_", " ")


# ============================================================================
# CLI helpers
# ============================================================================

def format_response(text: str) -> None:
    """Format response in CLI such that longer responses fit within the terminal without requiring
    the user to scroll.

    Args:
        text (str): Any longer text which may require formatting to be easily viewable.
    """
    print(textwrap.fill(text, width=100))


def newline_every_n_words(option: str, n: int) -> str:
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


def display_synthesis_request(synthesis_request: "SynthesisRequest") -> None:
    """Display the synthesis request after initial specification.
    
        Args:
            synthesis_request (SynthesisRequest): User preferences for data synthesis.
    """
    print(f"\nSample size mode: {synthesis_request.sample_size.mode}")

    if synthesis_request.sample_size.fixed_value is not None:
        print(f"Sample size: {synthesis_request.sample_size.fixed_value}")

    print(
        f"Privacy mode: {synthesis_request.privacy.mode}\n"
        f"Generator: {synthesis_request.generator.name}\n"
        f"Missing value handling: {synthesis_request.missing_values.mode}"
    )

    if synthesis_request.missing_values.mode == "impute":
        print(
            f"- Imputer: {synthesis_request.missing_values.impute.method}\n"
            "- Categorical strategy: "
            f"{format_option(synthesis_request.missing_values.impute.categorical_strategy)}\n"
            "- Numerical strategy: "
            f"{format_option(synthesis_request.missing_values.impute.numerical_strategy)}"
        )


# ============================================================================
# UI helpers
# ============================================================================

def display_synthesis_request_st(synthesis_request: "SynthesisRequest") -> None:
    """Display the synthesis request after initial specification (UI).

    Args:
        synthesis_request (SynthesisRequest): User preferences for data synthesis.
    """
    st.write(f"\n**Sample size mode:** {format_option(synthesis_request.sample_size.mode)}")

    if synthesis_request.sample_size.fixed_value is not None:
        st.write(f"**Sample size:** {synthesis_request.sample_size.fixed_value}")

    st.write(
        f"**Privacy mode:** {format_option(synthesis_request.privacy.mode)}\n\n"
        f"**Generator:** {synthesis_request.generator.name}\n\n"
        f"**Missing value handling:** {synthesis_request.missing_values.mode}"
    )

    if synthesis_request.missing_values.mode == "impute":
        st.write(
            f"- **Imputer:** {synthesis_request.missing_values.impute.method}\n\n"
            "- **Categorical strategy:** "
            f"{format_option(synthesis_request.missing_values.impute.categorical_strategy)}\n\n"
            "- **Numerical strategy:** "
            f"{format_option(synthesis_request.missing_values.impute.numerical_strategy)}"
        )


def page_navigation(back_page: str, next_page: str, next_disabled: bool = False,):
    """Page navigation buttons to either go back or forward in the workflow.

    Args:
        back_page (str): The previous page which can be revisited.
        next_page (str): The next page which can be visited.
        next_disabled (bool, optional): Disables ability to visit the next page. Defaults to False.
    """
    st.markdown("""
        <style>
        div[data-testid="stPageLink"] a {
            border: 1px solid #D3D3D3;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            text-align: center;
            display: block;
            text-decoration: none;
        }
        </style>
    """, unsafe_allow_html=True,)

    col1, _, _, _, _, _, col2 = st.columns(7)
    with col1:
        st.page_link(back_page, label="⟵", use_container_width=True,)

    with col2:
        st.page_link(
            next_page,
            label="⟶",
            use_container_width=True,
            disabled=next_disabled,
        )


def df_formatter_st(df: pd.DataFrame | Styler) -> None:
    """Format dataframes with consistent sizing when over 10 rows are present.

    Args:
        df (pd.DataFrame): The input dataframe.
    """
    if isinstance(df, Styler):
        n_rows = len(df.data)
    else:
        n_rows = len(df)

    if n_rows > 10:
        st.dataframe(df, hide_index=True, width="stretch", height=400,)
        st.caption("Scroll to view all features.")
    else:
        st.dataframe(df, hide_index=True, width="stretch",)


def info_helper_message(message: str, context: str, key: str,) -> None:
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


def display_workflow_progress(text: str) -> None:
    """Display a persistent workflow indicator in the Streamlit header.

    Args:
        text (str): Workflow progress text to display.
    """
    # Used to prevent sticky header from blocking interactions with Streamlit icons.
    st.html("""
    <style>
        [class*="st-key-workflow-header"] {
            pointer-events: none !important;
        }

        [class*="st-key-workflow-header"] * {
            pointer-events: none !important;
        }
    </style>
    """)

    with st_yled.sticky_header(
        height="56px",
        background_color="transparent",
        vertical_alignment="center",
        horizontal_alignment="center",
        padding="0px 16px",
        key="workflow-header",
    ):
        st.write(text)


def colorize_multiselect(color: str = "#56EB25") -> None:
    """Change the background color of selected multiselect tags.

    Args:
        color (str): CSS color for selected tags.
    """
    st.html(f"""
    <style>
        .stMultiSelect [data-tag] {{
            background-color: {color} !important;
            color: white !important;
        }}
    </style>
    """)


# ============================================================================
# Agent helpers
# ============================================================================

def parse_json(
    plan: str,
    model_cls: type[T],
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
    root: bool = False,
    ui: bool = False,
) -> T | SemanticTypes:
    """If agent fails to correctly format JSON response, prompts to try again.

    Args:
        plan (str): The agent output.
        model_cls (type[T]): Pydantic model class used to validate JSON response.
        agent (Optional[CompiledStateGraph]): LangGraph agent.
        config (RunnableConfig): Configuration passed to the agent during invocation.
        root (bool, optional): Whether to return the root value (only necessary for 
            `semantic_type_handling()`). Defaults to False.
        ui (bool): Whether to display warning in Streamlit. Defaults to False (CLI).

    Returns:
        T: Validated instance of the specified Pydantic model.
    """
    try:
        result = model_cls.model_validate_json(plan)

    except Exception as error:
        message = "Agent response could not be parsed. Asking agent to repair the response..."

        if ui:
            with st.expander("Agent JSON response"):
                st.code(
                    "\n".join(
                        f"{i}: {line}"
                        for i, line in enumerate(plan.splitlines(), 1)
                    ),
                    language="json",
                )

            # st.warning(message)
            # with st.expander("Validation error details"):
            #     st.code(f"{type(error).__name__}: {error}")

        else:
            print(f"\n{message}\nError: {type(error).__name__}: {error}")

        repair_prompt = json_repair_prompt(error, plan, model_cls)

        repair_raw = agent.invoke(
            {"messages": [{"role": "system", "content": repair_prompt,}]},
            config=config,
        )

        repaired_plan = extract_output(repair_raw)

        result = model_cls.model_validate_json(repaired_plan)

    return result.root if root else result # root: semantic_type_handling has a different structure


def extract_output(raw_message: dict) -> str:
    """Extract content from the agent's invocation output

    Args:
        raw_message (dict): Raw agent invocation output.

    Returns:
        str: Agent message.
    """
    response = raw_message["messages"][-1].content

    # TODO: need to consider other models which do not use <think>
    if "</think>" in response:
        return response.split("</think>")[1].strip()

    return response.strip()


@overload
def invoke_agent(
    prompt: str,
    model_cls: type[T],
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
    root: Literal[False] = False,
    ui: bool = False,
) -> T: ...


@overload
def invoke_agent(
    prompt: str,
    model_cls: type[T],
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
    root: Literal[True],
    ui: bool = False,
) -> SemanticTypes: ...


def invoke_agent(
    prompt: str,
    model_cls: type[T],
    agent: Optional[CompiledStateGraph],
    config: RunnableConfig,
    root: bool = False,
    ui: bool = False,
) -> T | SemanticTypes:
    """Invoke the agent and parse its response into the specified model.

    Args:
        prompt (str): Message sent to the agent.
        model_cls (type[T]): Pydantic model class used to validate JSON response.
        agent (Optional[CompiledStateGraph]): LangGraph agent.
        config (RunnableConfig): Configuration passed to the agent during invocation.
        root (bool): Whether to return the root value (only necessary for 
            `semantic_type_handling()`). Defaults to False.
        ui (bool): Whether to display warning in Streamlit. Defaults to False (CLI).

    Returns:
        T: Validated instance of the specified Pydantic model.
    """
    raw_response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt,}]},
        config=config,
    )

    plan = extract_output(raw_response)

    return parse_json(plan=plan, model_cls=model_cls, agent=agent, config=config, root=root, ui=ui,)


# ============================================================================
# Generator helpers
# ============================================================================

def output_default_parameters(generator_name: GeneratorNameAuto) -> dict[str, Any] | None:
    """Takes in generator name and outputs its default parameters. Used in select preferences
    and other instances where parameters are set to default.

    Args:
        generator_name (GeneratorNameAuto): Selected generator.

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

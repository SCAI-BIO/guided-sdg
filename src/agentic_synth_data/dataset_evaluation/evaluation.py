from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
import pyinputplus as pyip
import streamlit as st

from agentic_synth_data.schemas.attack_model import AttackMetrics
from agentic_synth_data.schemas.eval_model import AgentEval
from agentic_synth_data.schemas.syndat_model import SyndatMetrics
from agentic_synth_data.utils import format_response, extract_output, info_helper_message, \
    invoke_agent
from agentic_synth_data.prompts.evaluation_prompt import eval_review_message, evaluation_prompt
from agentic_synth_data.preference_setup import SynthesisRequest


def evaluation(
    synthesis_request: SynthesisRequest,
    syndat_metrics: SyndatMetrics,
    attack_metrics: AttackMetrics,
    agent: Optional[CompiledStateGraph] | None = None,
    config: RunnableConfig | None = None,
) -> AgentEval:
    """The agent evaluates the synthetic dataset using calculated metrics.

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
            missing value markers replaced with nan (norm_na_df).
        synthetic_df (pd.DataFrame): The synthetically generated dataframe.
        synthesis_request (SynthesisRequest): Dictionary of the user's preferences. This is used as
            a record of the processing done to generate the synthetic data.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to evaluate the
            synthetic data based on SYNDAT metrics. Defaults to None.
        config (RunnableConfig | None, optional): Configuration passed to the agent during
            invocation. Defaults to None.

    Returns:
        AgentEval: The decision plan based on the agent's assessment.
    """
    print("\nAgent is evaluating the results...")

    eval_plan = invoke_agent(
        prompt=evaluation_prompt(
            synthesis_request=synthesis_request,
            syndat_metrics=syndat_metrics,
            attack_metrics=attack_metrics,
        ),
        model_cls=AgentEval,
        agent=agent,
        config=config,
    )

    print("\nEvaluation:")
    print(f"- Status: {eval_plan.status}")
    if len(eval_plan.problem_columns) > 0:
        print(f"- Problem columns: {eval_plan.problem_columns}")
    print("- Recommendation:")
    format_response(eval_plan.recommendation)
    print("- Rationale:")
    format_response(eval_plan.rationale)
    return eval_plan


def evaluation_ui(
    synthesis_request: SynthesisRequest,
    syndat_metrics: SyndatMetrics,
    attack_metrics: AttackMetrics,
    agent: Optional[CompiledStateGraph] | None = None,
    config: RunnableConfig | None = None,
) -> AgentEval:
    """The agent evaluates the synthetic dataset using calculated metrics (UI).

    Args:
        real_df (pd.DataFrame): The original (data_path) dataframe or the original dataframe with 
            missing value markers replaced with nan (norm_na_df).
        synthetic_df (pd.DataFrame): The synthetically generated dataframe.
        synthesis_request (SynthesisRequest): Dictionary of the user's preferences. This is used as
            a record of the processing done to generate the synthetic data.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to evaluate the
            synthetic data based on SYNDAT metrics. Defaults to None.
        config (RunnableConfig | None, optional): Configuration passed to the agent during
            invocation. Defaults to None.

    Returns:
        AgentEval: The decision plan based on the agent's assessment.
    """
    if "agent_evaluation" not in st.session_state:
        with st.spinner("Agent is evaluating the results..."):
            eval_plan = invoke_agent(
                prompt=evaluation_prompt(
                    synthesis_request=synthesis_request,
                    syndat_metrics=syndat_metrics,
                    attack_metrics=attack_metrics,
                ),
                model_cls=AgentEval,
                agent=agent,
                config=config,
                ui=True,
            )

        st.session_state.agent_evaluation = eval_plan

    else:
        eval_plan = st.session_state.agent_evaluation

    info_helper_message(
        message="The agent has finished its evaluation of the synthetic dataset.",
        context=(
            "The agent evaluated the synthetic dataset based on:\n"
            "- Your preferences.\n"
            "- Calculated syndat metrics.\n"
            "- Calculated Anonymeter metrics."
        ),
        key="agent_eval_synth_data_info",
    )

    display_agent_evaluation_ui(agent_decision=eval_plan,)

    return eval_plan


def display_agent_evaluation_ui(agent_decision: AgentEval) -> None:
    """Display the agent's evaluation (UI).

    Args:
        agent_decision (AgentEval): The decision plan based on the agent's assessment.
    """
    st.subheader("Evaluation:")
    st.write(f"- **Status:** {agent_decision.status}")
    if len(agent_decision.problem_columns) > 0:
        st.write(f"- **Problem columns:** {agent_decision.problem_columns}")
    st.write(
        "#### - Recommendation:\n"
        f"{agent_decision.recommendation}\n"
        "#### - Rationale:\n"
        f"{agent_decision.rationale}"
    )


def evaluation_review(
    agent_decision: AgentEval,
    agent: Optional[CompiledStateGraph] | None = None,
    config: RunnableConfig | None = None
) -> str:
    """User asks questions about agent's assessment/discusses details to focus on.

    Args:
        agent_decision (AgentEval): Agent's evaluation decision.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to review the 
                                                               evaluation. Defaults to None.
        config (RunnableConfig | None, optional): Configuration passed to the agent during
            invocation. Defaults to None.

    Returns:
        str: Whether the workflow should finish or be revised.
    """
    confirm = pyip.inputYesNo("\nDo you have questions about the evaluation results?\n")

    while confirm == "yes":
        response = pyip.inputStr("\nEnter your question/concern:\n")
        print("\nAgent is preparing its response...")
        prompt = eval_review_message(agent_decision, response)
        raw_response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt,}]},
            config=config,
        )

        format_response(extract_output(raw_response))

        confirm = pyip.inputYesNo("\nDo you have another question?\n")

    print(f"\nWorkflow result status: {agent_decision.status}")
    choice = pyip.inputMenu(
        ["finish", "revise"],
        "\nWould you like to finish or revise the workflow?\n", 
        numbered=True
    )

    return choice


def evaluation_review_ui(
    agent_decision: AgentEval,
    agent: Optional[CompiledStateGraph] | None = None,
    config: RunnableConfig | None = None,
    version_id: int | None = None,
) -> str:
    """User asks questions about agent's assessment/discusses details to focus on (UI).

    Args:
        agent_decision (AgentEval): Agent's evaluation decision.
        agent (Optional[CompiledStateGraph] | None, optional): LangGraph agent used to review the 
                                                               evaluation. Defaults to None.
        config (RunnableConfig | None, optional): Configuration passed to the agent during
            invocation. Defaults to None.

    Returns:
        str: Whether the workflow should finish or be revised.
    """
    if "evaluation_review_history" not in st.session_state:
        st.session_state.evaluation_review_history = []

    confirm = st.selectbox(
        "Do you have questions about the evaluation results?",
        ["yes", "no"],
        index=None,
        key=f"evaluation_review_confirm_{version_id}",
    )

    if confirm is None:
        st.stop()

    if confirm == "yes":
        st.caption("Questions about the evaluation")
        with st.container(border=True, height=350):
            for item in st.session_state.evaluation_review_history:
                with st.chat_message("user"):
                    st.write(item["question"])

                with st.chat_message("assistant"):
                    st.write(item["answer"])

            new_question = st.chat_input(
                "Enter your question/concern.",
                key=f"evaluation_review_input_{version_id}",
            )

            if new_question:
                with st.spinner("\nAgent is preparing its response..."):
                    prompt = eval_review_message(agent_decision, new_question)
                    raw_response = agent.invoke(
                        {"messages": [{"role": "user", "content": prompt,}]},
                        config=config,
                    )

                    answer = extract_output(raw_response)

                st.session_state.evaluation_review_history.append({
                    "question": new_question,
                    "answer": answer,
                })

                st.rerun()

        finished = st.button(
            "Done asking questions",
            key=f"evaluation_review_done_{version_id}",
        )

        if not finished:
            st.stop()

    # If no is selected:
    st.write(f"Workflow result status: {agent_decision.status}")

    choice = st.selectbox(
        label="Would you like to finish or revise the workflow?",
        options=["finish", "revise"],
        index=None,
        key=f"evaluation_workflow_choice_{version_id}",
    )

    if choice is None:
        st.stop()

    return choice

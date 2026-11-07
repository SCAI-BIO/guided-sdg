import copy
from contextlib import redirect_stdout
from pathlib import Path
from collections.abc import Hashable
import os

from langchain_core.runnables import RunnableConfig
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model, BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from agentic_synth_data.dataset_evaluation.evaluation import display_agent_evaluation_ui, \
    evaluation, evaluation_ui, evaluation_review_ui, evaluation_review
from agentic_synth_data.dataset_evaluation.attack_metrics import calculate_attack_metrics, \
    calculate_attack_metrics_ui, display_attack_metrics_ui, ANONYMETER_METRIC_DESCRIPTION
from agentic_synth_data.report import write_report
from agentic_synth_data.dataset_evaluation.syndat_metrics import calculate_metrics, \
    calculate_metrics_ui, generate_plots, generate_plots_ui, display_metrics_ui
from agentic_synth_data.dataset_preparation.dtype_handling import handle_dtypes, handle_dtypes_ui, \
    missing_value_precheck, missing_value_precheck_ui
from agentic_synth_data.dataset_preparation.load_dataframe import load_df
from agentic_synth_data.dataset_preparation.semantic_type_handling import handle_semantic_types, \
    handle_semantic_types_ui, prepare_for_syndat
from agentic_synth_data.dataset_preparation.missing_value_handling import handle_missing_values, \
    handle_missing_values_ui
from agentic_synth_data.preference_setup import select_preferences, select_preferences_ui
from agentic_synth_data.prompts.system_prompt import SYSTEM_PROMPT
from agentic_synth_data.report import Report, ReviewReport
from agentic_synth_data.rerun_workflow import plan_update, plan_update_ui
from agentic_synth_data.schemas.models import SynthesisRequest, ImputationConfig
from agentic_synth_data.schemas.eval_model import AgentEval
from agentic_synth_data.schemas.streamlit_helpers import StreamlitOutput
from agentic_synth_data.strategies.factory import get_generator_strategy
from agentic_synth_data.synthetic_generation.generation import select_generator, \
    select_generator_ui, generator_profile, generator_profile_ui
from agentic_synth_data.utils import DATA_DIR, OUTPUT_DIR, SemanticTypes, df_formatter_st, \
    format_response, output_default_parameters, comma_sep, step


class DataGen:
    """Manage an agent-assisted workflow for synthetic tabular data generation.

    Handles data preparation, generator selection, synthesis, and evaluation of fidelity, utility,
    and privacy. Supports iterative workflow revisions based on evaluation results and user feedback
    through CLI and Streamlit interfaces.
    """
    # ============================================================================
    # Initialization and configuration
    # ============================================================================

    def __init__(self, data_path):
        # Workflow identification and revision state
        self.data_path: str = data_path
        self.version_id = 1
        self.parent_version = None
        self.pending_revision_plan: dict | None = None

        # Workflow decisions and metadata
        self.synthesis_request: SynthesisRequest | None = None
        self.python_dtypes: dict[Hashable, str] | None = None
        self.semantic_types: SemanticTypes | None = None
        self.agent_decision: AgentEval | None = None

        # Dataframes
        self.source_df: pd.DataFrame | None = None # Input data (optional: with marker replacement)
        self.selected_df: pd.DataFrame | None = None # Handled dtypes with specified covariates

        self.train: pd.DataFrame | None = None # Train split of selected
        self.test: pd.DataFrame | None = None # Test split of selected (used in Anonymeter)

        self.prepared_df: pd.DataFrame | None = None # Training data prepared for generator
        self.prepared_test: pd.DataFrame | None = None # Prepared testing data

        self.synthetic_df: pd.DataFrame | None = None # Synthetic data

        # Generated file paths
        self.metadata_path: Path | None = None
        self.prepared_path: Path | None = None # What is input into the tool (either with or without missing value handling)
        self.synthetic_path: Path | None = None

        # Agent configuration
        self.model: BaseChatModel = self.create_model()
        self.checkpointer: InMemorySaver = InMemorySaver()
        self.config: RunnableConfig = {"configurable": {"thread_id": 1}}
        self.agent: CompiledStateGraph = self.create_agent()

        # Workflow report
        self.report: Report = Report()


    def create_model(self) -> BaseChatModel:
        """Initialize the selected large language model."""
        model = init_chat_model(
            "deepseek-ai/DeepSeek-V4.1-Flash",
            model_provider="openai",
            base_url=os.environ.get("BASE_URL"),
        )
        return model


    def create_agent(self) -> CompiledStateGraph:
        """Initialize agent with the desired model for use in the workflow."""
        agent = create_agent(
            model=self.model,
            checkpointer=self.checkpointer,
            system_prompt=SYSTEM_PROMPT,
        )
        return agent


    # ============================================================================
    # Step 1: Load dataset
    # ============================================================================

    def load_dataframe(self) -> pd.DataFrame:
        """Step 1: Setup source dataframe"""
        self.source_df = load_df(self.data_path)
        self.report.input.path = self.data_path
        self.report.input.original_shape = self.source_df.shape
        self.report.input.version_id = self.version_id
        self.report.input.parent_version = self.parent_version
        return self.source_df


    # ============================================================================
    # Step 2: Apply preferences
    # ============================================================================

    def apply_preferences(self):
        """Step 2: Setup preferences"""
        step(name="Select preferences", space=False,)
        self.synthesis_request = select_preferences()
        return self.synthesis_request


    def apply_preferences_ui(self):
        """Step 2: Setup preferences (UI)
        """
        request = select_preferences_ui()

        if request is None:
            st.stop()

        self.synthesis_request = request
        return self.synthesis_request


    # ============================================================================
    # Step 3: Preprocess dataset and select generator
    # ============================================================================

    def preprocess_dtypes(self) -> pd.DataFrame:
        """Step 3: Preprocess
            i. Check for missing value markers
            ii. Handle dtypes (both categories)
            iii. Select covariates and split the dataset (pending)
        """
        # i. Check for missing value markers
        step(name="Normalize missing values")

        missing_report = missing_value_precheck(self.source_df.copy())

        self.report.prep.missing_markers_detected = missing_report.detected_markers
        self.report.prep.missing_markers_replaced = missing_report.replaced_markers

        # Save result whether missing markers have been converted or not.
        self.source_df = missing_report.df.copy()
        print("\n⇨ Dataframe has been saved for use during evaluation.")

        # ii. Handle dtypes (both categories)
        step(name="Handle data types")
        dtype_handled_df, self.python_dtypes = handle_dtypes(self.source_df.copy())

        step(name="Handle semantic types")
        self.semantic_types = handle_semantic_types(
            df=dtype_handled_df.copy(), agent=self.agent, config=self.config
        )

        # iii. Select covariates and split the dataset (pending)
        # TODO: Implement column selection here.
        self.selected_df = dtype_handled_df.copy()

        # Update stypes and dtypes to exclude selected cols
        self.semantic_types = {col: self.semantic_types[col] for col in self.selected_df.columns}
        self.python_dtypes = {col: self.python_dtypes[col] for col in self.selected_df.columns}

        self.report.prep.dtypes = self.python_dtypes
        self.report.prep.stypes = self.semantic_types

        self.train, self.test = train_test_split(self.selected_df, test_size=0.2, random_state=42,)
        print("\n⇨ Created train/test split")
        return self.train


    def preprocess_dtypes_ui(self) -> pd.DataFrame:
        """Step 3: Preprocess (UI)
            i. Check for missing value markers
            ii. Handle dtypes (both categories)
            iii. Select covariates and split the dataset
        """
        # i. Check for missing value markers
        st.subheader("Normalize missing values")
        st.write(
            "**Look for any missing value markers that should be encoded as null values within "
            "the dataset prior to analysis.**"
        )
        missing_report = missing_value_precheck_ui(self.source_df.copy())

        if missing_report is None:
            st.stop()

        self.report.prep.missing_markers_detected = missing_report.detected_markers
        self.report.prep.missing_markers_replaced = missing_report.replaced_markers

        self.source_df = missing_report.df.copy()
        st.info("⇨ Dataframe has been saved for use during evaluation.")

        st.write("---")

        # ii. Handle dtypes (both categories)
        st.subheader("Handle data types")
        st.write(
            "Automatically detect the majority Python data types present for each variable "
            "in the dataframe."
        )
        dtype_result = handle_dtypes_ui(self.source_df.copy())

        if dtype_result is None:
            st.stop()

        dtype_handled_df, self.python_dtypes = dtype_result

        st.write("---")

        st.subheader("Handle semantic types")
        st.write(
            "Have the agent determine the semantic meaning of each variable based on "
            "information found in the dataset."
        )

        stype_result = handle_semantic_types_ui(
            df=dtype_handled_df.copy(), agent=self.agent, config=self.config,
        )

        self.semantic_types = stype_result

        st.write("---")
        st.subheader("Final data types present in each column")

        final_dtypes = pd.DataFrame({
            "Feature": self.semantic_types.keys(),
            "Data type": [self.python_dtypes[col] for col in self.semantic_types.keys()],
            "Semantic type": self.semantic_types.values(),
        })

        df_formatter_st(final_dtypes)

        st.write("---")

        # iii. Select covariates and split the dataset
        # TODO: also must consider when missing value handling is applied.
        st.subheader("Covariate exclusion")

        st.write(
            "Select the features that should be retained for synthetic data generation. "
            "Unselected features will be excluded."
        )

        cols = dtype_handled_df.columns.tolist()

        selected_cols = st.multiselect(
            label="Select features:",
            options=cols,
            default=cols,
            key="selected_covariates",
        )

        if not selected_cols:
            st.warning("Select at least one column.")
            st.stop()

        st.write(
            f"The following **{len(selected_cols)}** features will be retained: "
            f"{comma_sep(selected_cols)}"
        )

        finished = st.button(label="Continue with the current selection.",)

        if not finished:
            st.stop()

        self.selected_df = dtype_handled_df[selected_cols].copy()

        # Update stypes and dtypes to exclude selected cols
        self.semantic_types = {col: self.semantic_types[col] for col in self.selected_df.columns}
        self.python_dtypes = {col: self.python_dtypes[col] for col in self.selected_df.columns}

        self.report.prep.dtypes = self.python_dtypes
        self.report.prep.stypes = self.semantic_types

        self.train, self.test = train_test_split(self.selected_df, test_size=0.2, random_state=42,)

        st.info(
            f"⇨ Created test/train split: {len(self.train)} training rows and {len(self.test)} "
            "testing rows."
        )
        return self.train


    def generator_selection(self):
        """Step 3: Preprocess
            iii. Select/verify chosen generator
        """
        step(name="Select data generator")
        print()
        profile = generator_profile(self.train.copy())
        self.report.prep.profile = profile
        self.synthesis_request = select_generator(
            synthesis_request=self.synthesis_request,
            semantic_types=self.semantic_types,
            profile=profile,
            agent=self.agent,
            config=self.config,
        )
        return self.synthesis_request


    def generator_selection_ui(self):
        """Step 3: Preprocess
            iii. Select/verify chosen generator
        """
        profile = generator_profile_ui(self.train.copy())
        self.report.prep.profile = profile

        synthesis_request = select_generator_ui(
            synthesis_request=self.synthesis_request,
            semantic_types=self.semantic_types,
            profile=profile,
            agent=self.agent,
            config=self.config,
        )

        if synthesis_request is None:
            st.stop()

        self.synthesis_request = synthesis_request
        return self.synthesis_request


    def preprocess_missing_vals(self):
        """Step 3: Preprocess
            iv. Handle missing values (if needed for the generator)
        """
        self.report.prep.missing_before = self.selected_df.isna().sum().sum()

        mechanism = self.synthesis_request.generator.name

        if mechanism in ["PATEGAN", "CTGAN",]: # NOTE: For when missing handing is needed.
            step(name="Handle missing values")

            result = handle_missing_values(
                train=self.train.copy(),
                test=self.test.copy(),
                synthesis_request=self.synthesis_request,
                semantic_types=self.semantic_types,
                agent=self.agent,
                config=self.config,
            )

            if result is not None:
                self.prepared_df, self.prepared_test, self.synthesis_request = result

        else:
            self.prepared_df = self.train.copy()
            self.prepared_test = self.test.copy()

            print(
                "\nThe selected data generator does not require missing value handling: "
                f"{mechanism}"
            )

            # Reset missing value configuration
            if self.synthesis_request.missing_values.mode == "impute":
                self.synthesis_request.missing_values.impute = ImputationConfig()

            elif self.synthesis_request.missing_values.impute == "threshold":
                self.synthesis_request.missing_values.threshold = None

            self.synthesis_request.missing_values.mode = None

        self.report.input.prepared_shape = self.prepared_df.shape
        self.report.prep.missing_after = self.prepared_df.isna().sum().sum()

        return self.prepared_df, self.synthesis_request


    def preprocess_missing_vals_ui(self):
        """Step 3: Preprocess
            iv. Handle missing values (if needed for the generator) (UI)
        """
        self.report.prep.missing_before = self.selected_df.isna().sum().sum()

        mechanism = self.synthesis_request.generator.name

        if mechanism in ["PATEGAN", "CTGAN",]: # NOTE: For when missing handing is needed.
            st.header("Handle missing values")

            result = handle_missing_values_ui(
                train=self.train.copy(),
                test=self.test.copy(),
                synthesis_request=self.synthesis_request,
                semantic_types=self.semantic_types,
                agent=self.agent,
                config=self.config,
            )

            if result is not None:
                self.prepared_df, self.prepared_test, self.synthesis_request = result

        else:
            self.prepared_df = self.train.copy()
            self.prepared_test = self.test.copy()

            st.info(
                f"The selected data generator does not require missing value handling: {mechanism}."
            )

            # Reset missing value configuration
            if self.synthesis_request.missing_values.mode == "impute":
                self.synthesis_request.missing_values.impute = ImputationConfig()

            elif self.synthesis_request.missing_values.impute == "threshold":
                self.synthesis_request.missing_values.threshold = None

            self.synthesis_request.missing_values.mode = None

        self.report.input.prepared_shape = self.prepared_df.shape
        self.report.prep.missing_after = self.prepared_df.isna().sum().sum()

        return self.prepared_df, self.synthesis_request


    # ============================================================================
    # Step 4: Prepare generator inputs
    # ============================================================================

    # TODO combine CLI and UI functions with minimal changes (only print statements have been changed)
    def save_prepared_data(self) -> Path:
        """ Step 4: Setup tool data
            i. Save the prepared data
        """
        input_path = Path(self.data_path)

        # The prepared data for generator tool input
        output_filename = f"{input_path.stem}_prepared.csv"
        output_dir = DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        self.prepared_path = output_dir / output_filename
        self.prepared_df.to_csv(self.prepared_path, index=False)

        self.report.gen.prep_path = self.prepared_path
        return self.prepared_path


    def save_prepared_data_cli(self) -> Path:
        """ Step 4: Setup tool data
            i. Save the prepared data (CLI)
        """
        step(name="Create metadata for tool use")

        prepared_path = self.save_prepared_data()

        format_response(f"- Created csv for generator input: {prepared_path}")
        return prepared_path


    def save_prepared_data_ui(self) -> Path:
        """Step 4: Setup tool data
            i. Save the prepared data (UI)
        """
        prepared_path = self.save_prepared_data()

        st.write(f"- Created csv for generator input: `{self.prepared_path}`")
        return prepared_path


    def create_metadata(self):
        """Step 4: Setup tool data
            ii. Create metadata for tool use
        """
        strategy = get_generator_strategy(self.synthesis_request.generator.name)

        self.metadata_path = strategy.prepare_metadata(
            prep_path=self.prepared_path,
            semantic_types=self.semantic_types,
        )


    # ============================================================================
    # Step 5: Generate synthetic data
    # ============================================================================

    def run_synthesis(self):
        """Step 5: Synthesize data"""
        step(name="Synthesize synthetic data")
        strategy = get_generator_strategy(self.synthesis_request.generator.name)

        # Set sample size
        if self.synthesis_request.sample_size.mode == "fixed_value":
            sample_size = self.synthesis_request.sample_size.fixed_value
        else:
            sample_size = len(self.source_df)

        self.synthetic_path, self.synthetic_df = strategy._run(
            n_samples=sample_size,
            prep_path=self.prepared_path,
            parameters=self.synthesis_request.generator.parameters,
            version_id=self.version_id,
        )

        self.report.gen.synth_path = self.synthetic_path
        self.report.gen.synth_shape = self.synthetic_df.shape
        self.report.gen.params = self.synthesis_request.generator.parameters
        return self.synthesis_request, self.synthetic_path, self.synthetic_df


    def run_synthesis_ui(self):
        """Step 5: Synthesize data (UI)"""
        if st.session_state.get(f"generation_complete_{self.version_id}", False):
            self.synthetic_path = st.session_state.synthetic_path
            self.synthetic_df = st.session_state.synthetic_df

        else:
            generator = self.synthesis_request.generator.name
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            strategy = get_generator_strategy(generator_name=generator)

            # Set sample size
            if self.synthesis_request.sample_size.mode == "fixed_value":
                sample_size = self.synthesis_request.sample_size.fixed_value
            else:
                sample_size = len(self.source_df)

            with st.status(f"Generating data with {generator}...", expanded=True) as status:
                output = StreamlitOutput()

                with redirect_stdout(output):
                    self.synthetic_path, self.synthetic_df = strategy._run(
                        n_samples=sample_size,
                        prep_path=self.prepared_path,
                        parameters=self.synthesis_request.generator.parameters,
                        version_id=self.version_id,
                    )

                status.update(
                    label=f"⇨ Synthetic generation completed with {generator} (click to view " \
                        "terminal outputs).",
                    state="complete",
                    expanded=False,
                )

            st.session_state.synthetic_path = self.synthetic_path
            st.session_state.synthetic_df = self.synthetic_df
            st.session_state[f"generation_complete_{self.version_id}"] = True

            self.report.gen.synth_path = self.synthetic_path
            self.report.gen.synth_shape = self.synthetic_df.shape
            self.report.gen.params = self.synthesis_request.generator.parameters

        df_formatter_st(self.synthetic_df.copy().head())
        st.caption(
            "**Note:** Only the first five rows of the synthetic dataset are displayed on this "
            "page."
        )
        return self.synthesis_request, self.synthetic_path, self.synthetic_df


    # ============================================================================
    # Step 6: Evaluate synthetic data
    # ============================================================================

    def evaluate(self):
        """Step 6: Evaluate results"""
        step(name="Evaluate synthetic data")

        real = self.selected_df.copy()
        synth = self.synthetic_df.copy()

        real, synth = prepare_for_syndat(
            real=real,
            synth=synth,
            stypes=self.semantic_types,
        )

        syndat_metrics = calculate_metrics(real=real, synth=synth,)

        self.report.evaluate.syndat = syndat_metrics

        attack_metrics = calculate_attack_metrics(
            train=self.train,
            test=self.test,
            synth=synth,
            stypes=self.semantic_types,
            dtypes=self.python_dtypes,
            agent=self.agent,
            config=self.config,
        )

        self.report.evaluate.attack = attack_metrics

        self.report.evaluate.plots = generate_plots(
            real=real, synth=synth, version_id=self.version_id,
        )

        self.agent_decision = evaluation(
            synthesis_request=self.synthesis_request,
            syndat_metrics=syndat_metrics,
            attack_metrics=attack_metrics,
            agent=self.agent,
            config=self.config,
        )

        self.report.review = ReviewReport(
            problem_columns=self.agent_decision.problem_columns,
            status=self.agent_decision.status,
            recommendation=self.agent_decision.recommendation,
            rationale=self.agent_decision.rationale,
        )

        write_report(
            report=self.report,
            synthesis_request=self.synthesis_request,
            data_path=self.data_path,
            version_id=self.version_id,
        )


    def evaluate_ui(self):
        """Step 6: Evaluate results"""
        synth = self.synthetic_df.copy()
        if self.norm_na_df is not None:
            real = self.norm_na_df.copy()
        else:
            real = self.source_df.copy()

        real = real[self.selected_df.columns].copy()

        real, synth = prepare_for_syndat(real=real, synth=synth, stypes=self.semantic_types,)

        col1, col2 = st.columns(2)

        # Retrieve metrics from the previous version (if available)
        prev_syndat_metrics, prev_attack_metrics = self.get_prev_evaluation_metrics()

        with col1:
            syndat_metrics = calculate_metrics_ui(
                real=real, synth=synth, prev_metrics=prev_syndat_metrics,
            )

            self.report.evaluate.syndat = syndat_metrics

        attack_key = f"attack_metrics_{self.version_id}"

        # Anonymeter metrics take longer to load, so they are saved for display upon page reload.
        with col2:
            if attack_key not in st.session_state:
                attack_metrics = calculate_attack_metrics_ui(
                    train=self.train,
                    test=self.test,
                    synth=synth,
                    stypes=self.semantic_types,
                    dtypes=self.python_dtypes,
                    key_prefix=f"calculation_v{self.version_id}",
                    prev_metrics=prev_attack_metrics,
                    agent=self.agent,
                    config=self.config,
                )

                st.session_state[attack_key] = attack_metrics

            else:
                attack_metrics = st.session_state[attack_key]

                display_attack_metrics_ui(
                    attack_metrics=attack_metrics,
                    description=ANONYMETER_METRIC_DESCRIPTION,
                    key_prefix=f"evaluation_v{self.version_id}",
                    prev_metrics=prev_attack_metrics,
                )

        self.report.evaluate.attack = attack_metrics

        st.write("---")

        st.subheader("View syndat plots")

        with st.spinner("Loading plots..."):
            self.report.evaluate.plots = generate_plots_ui(
                real=real, synth=synth, version_id=self.version_id,
            )

        st.write("---")

        # Display agent evaluation of results
        self.agent_decision = evaluation_ui(
            synthesis_request=self.synthesis_request,
            syndat_metrics=syndat_metrics,
            attack_metrics=attack_metrics,
            agent=self.agent,
            config=self.config,
        )

        # Write report
        self.report.review = ReviewReport(
            problem_columns=self.agent_decision.problem_columns,
            status=self.agent_decision.status,
            recommendation=self.agent_decision.recommendation,
            rationale=self.agent_decision.rationale,
        )

        write_report(
            report=self.report,
            synthesis_request=self.synthesis_request,
            data_path=self.data_path,
            version_id=self.version_id,
        )

        data_name = Path(self.data_path.stem)

        report_dir = (
            OUTPUT_DIR / f"version_{self.version_id}" / f"report_{data_name}_v{self.version_id}.md"
        )

        st.info(
            f"⇨ Created a report for version_{self.version_id}: `{report_dir}`"
        )


    # ============================================================================
    # Post-evaluation review
    # ============================================================================

    def post_evaluation_check(self):
        """Revise: Check in with user and possibly change the plan"""
        step(name="Evaluate workflow")

        choice = evaluation_review(
            agent_decision=self.agent_decision, agent=self.agent, config=self.config,
        )

        if choice == "revise":
            selected_updates = plan_update(
                synthesis_request=self.synthesis_request,
                agent_decision=self.agent_decision,
                report=self.report,
                agent=self.agent,
                config=self.config,
            )

            if selected_updates is not None:
                for key, value in selected_updates.items():
                    if key == "update_params":
                        self.report.update.selected_plan_updates[key] = value
                    else:
                        self.report.update.selected_plan_updates[key] = value.model_dump()

                revised = self.create_revision(selected_updates)
                return revised

            return self

        return self


    def post_evaluation_check_ui(self):
        """Revise: Check in with user and possibly change the plan (UI)"""
        # Considering revision versions
        if "revision_versions" not in st.session_state:
            st.session_state.revision_versions = []

        if "revision_pending_from" not in st.session_state:
            st.session_state.revision_pending_from = None

        # Review interface
        review_container = st.empty()
        revision_container = st.empty()

        if st.session_state.revision_pending_from == self.version_id:
            review_container.empty()

            with revision_container.container():
                selected_updates = plan_update_ui(
                    synthesis_request=self.synthesis_request,
                    agent_decision=self.agent_decision,
                    report=self.report,
                    agent=self.agent,
                    config=self.config,
                    version_id=self.version_id,
                )

                if selected_updates is not None:
                    for key, value in selected_updates.items():
                        if key == "update_params":
                            self.report.update.selected_plan_updates[key] = value
                        else:
                            self.report.update.selected_plan_updates[key] = value.model_dump()

                    revised = self.create_revision_ui(selected_updates)

                    st.session_state.current_revision = revised
                    st.session_state.revision_history[revised.version_id] = revised
                    st.session_state.revision_pending_from = None

                    st.rerun()

            return self

        with review_container.container():
            choice = evaluation_review_ui(
                agent_decision=self.agent_decision,
                agent=self.agent,
                config=self.config,
                version_id=self.version_id,
            )

        if choice == "revise":
            next_version = self.version_id + 1

            if next_version not in st.session_state.revision_versions:
                st.session_state.revision_versions.append(next_version)

            st.session_state.revision_pending_from = self.version_id
            st.session_state.pending_active_revision_tab = f"Revision {next_version - 1}"

            # Clear the previous review interface
            review_container.empty()
            st.rerun()

        if choice == "finish":
            st.info("The workflow has been completed.")
            # TODO: redirect to download synthetic data for the preferred version
            # Click here to download the most recent synthetic dataset or to review actions committed in the report
            # show all (show hte same for the reports seciton)

        return self


    # ============================================================================
    # Revision management
    # ============================================================================

    def create_revision(self, update_plan:dict) -> "DataGen":
        """Copy datagen with the approved revisions."""
        print("\n" + "=" * 25 + "[REVISION: Create and update workflow copy]" + "=" * 25)
        revised = DataGen(self.data_path)

        revised.version_id = self.version_id + 1
        revised.parent_version = self.version_id

        revised.model = self.model
        revised.agent = self.agent
        revised.checkpointer = self.checkpointer
        revised.config = copy.deepcopy(self.config)

        revised.synthesis_request = self.synthesis_request.model_copy(deep=True)

        revised.source_df = self.source_df.copy()
        revised.selected_df = self.selected_df.copy()
        revised.train = self.train.copy()
        revised.test = self.test.copy()

        revised.python_dtypes = copy.deepcopy(self.python_dtypes)
        revised.semantic_types = copy.deepcopy(self.semantic_types)

        revised.report = copy.deepcopy(self.report)
        revised.report.input.version_id = revised.version_id
        revised.report.input.parent_version = revised.parent_version

        selection = revised._revision_scope(update_plan)

        revised._apply_revision_plan(update_plan)
        revised._reset_from_selection(selection)
        revised._rerun_selection(update_plan)
        return revised


    def create_revision_ui(self, update_plan:dict) -> "DataGen":
        """Copy datagen with the approved revisions (UI)."""
        st.write("\n" + "=" * 25 + "[REVISION: Create and update workflow copy]" + "=" * 25)
        revised = DataGen(self.data_path)

        revised.version_id = self.version_id + 1
        revised.parent_version = self.version_id

        revised.model = self.model
        revised.agent = self.agent
        revised.checkpointer = self.checkpointer
        revised.config = copy.deepcopy(self.config)

        revised.synthesis_request = self.synthesis_request.model_copy(deep=True)

        revised.source_df = self.source_df.copy()
        revised.selected_df = self.selected_df.copy()
        revised.train = self.train.copy()
        revised.test = self.test.copy()

        revised.python_dtypes = copy.deepcopy(self.python_dtypes)
        revised.semantic_types = copy.deepcopy(self.semantic_types)

        revised.report = copy.deepcopy(self.report)
        revised.report.input.version_id = revised.version_id
        revised.report.input.parent_version = revised.parent_version

        selection = revised._revision_scope(update_plan)

        revised._apply_revision_plan(update_plan)
        revised._reset_from_selection(selection)

        revised.pending_revision_plan = update_plan
        return revised


    def _revision_scope(self, update_plan: dict) -> str:
        """Indicates what has been selected for revision in `_reset_from_selection`."""
        has_generator = "update_mech" in update_plan
        has_parameter = "update_params" in update_plan
        has_missing = "update_missing" in update_plan

        if has_generator and has_parameter and has_missing:
            return "all"

        if has_generator and has_missing:
            return "generator_missing_selection"

        if has_generator and has_parameter:
            return "generator_parameter_selection"

        if has_parameter and has_missing:
            return "parameter_missing_selection"

        if has_generator:
            return "generator_selection"

        if has_parameter:
            return "parameter_revision"

        if has_missing:
            return "missing_value"

        raise ValueError("No updates were selected.")


    def _reset_from_selection(self, selection:str) -> None:
        """Clears previous outputs which are expected to be updated after changing the generator,
        missing value, or the selection of both options. Additionally, clears the report in the
        altered sections.
        """
        self.prepared_df = None
        self.prepared_test = None
        self.prepared_path = None
        self.metadata_path = None
        self.synthetic_df = None
        self.agent_decision = None

        # Covers "generator_selection"
        self.report.gen.prep_path = None
        self.report.gen.synth_path = None
        self.report.gen.synth_shape = None

        if selection in {
            "parameter_revision",
            "generator_parameter_selection",
            "parameter_missing_selection",
            "all", # Generator, parameter, and missing value handling selection.
        }:
            self.report.gen.params = None

        if selection in {
            "missing_value",
            "generator_missing_selection", # Generator and missing handling selection.
            "parameter_missing_selection",
            "all",
        }:
            self.report.prep.missing_before = {}
            self.report.prep.missing_after = {}

        self.report.review = ReviewReport()


    def _apply_revision_plan(self, update_plan:dict) -> None:
        """Uses the update_plan to update the synthesis request according to the specified revision
        information."""
        if "update_mech" in update_plan:
            option = update_plan["update_mech"]
            self.synthesis_request.generator.name = option.generator_name
            self.synthesis_request.generator.parameters = output_default_parameters(
                option.generator_name
            )

        if "update_params" in update_plan:
            self.synthesis_request.generator.parameters = update_plan["update_params"]

        if "update_missing" in update_plan:
            option = update_plan["update_missing"]
            self.synthesis_request.missing_values.mode = option.mode

            if option.mode == "impute":
                self.synthesis_request.missing_values.impute = ImputationConfig(
                    method=option.imputation_method,
                    categorical_strategy=option.categorical_strategy,
                    numerical_strategy=option.numerical_strategy,
                )

            else:
                self.synthesis_request.missing_values.impute = ImputationConfig()

    def _rerun_selection(self, update_plan: dict) -> pd.DataFrame:
        """Apply approved generator revision and rerun dependent workflow steps."""
        if "update_mech" in update_plan:
            print(f"\nNew generator: {update_plan['update_mech'].generator_name}")
            print("\nReason:")
            format_response(update_plan['update_mech'].reason)

        if "update_params" in update_plan:
            print(f"\nNew parameter configuration: {update_plan['update_params']}")

        if "update_missing" in update_plan:
            print(f"\nNew missing handling strategy: {update_plan['update_missing'].mode}")
            print(f"\nReason: {update_plan['update_missing'].reason}")

        self.report.update.selected_plan_updates = {}

        for key, value in update_plan.items():
            if key == "update_params":
                self.report.update.selected_plan_updates[key] = value
            else:
                self.report.update.selected_plan_updates[key] = value.model_dump()

        self.preprocess_missing_vals()
        self.save_prepared_data()
        self.create_metadata()
        self.run_synthesis()
        self.evaluate()
        final_gen = self.post_evaluation_check()
        return final_gen.synthetic_df


    def _rerun_selection_ui(self, update_plan: dict) -> pd.DataFrame:
        """Apply approved generator revision and rerun dependent workflow steps (UI)."""
        with st.container(border=True):
            st.success("Revision choices have been confirmed.")
            # TODO: show previous choices and what specifically was changed, even with an arrow.

            if "update_mech" in update_plan:
                generator_name = update_plan['update_mech'].generator_name

                # TODO: show previous generator if gneerator changed

                st.write(f"**Current generator:** {generator_name}\n\n**Reason:**\n")
                st.write(update_plan['update_mech'].reason)

            if "update_params" in update_plan:
                st.write(f"**New parameter configuration:** {update_plan['update_params']}")

            if "update_missing" in update_plan:
                st.write(
                    f"**New missing handling strategy:** {update_plan['update_missing'].mode}\n\n"
                    f"**Reason:** {update_plan['update_missing'].reason}"
                )

        self.report.update.selected_plan_updates = {}

        for key, value in update_plan.items():
            if key == "update_params":
                self.report.update.selected_plan_updates[key] = value
            else:
                self.report.update.selected_plan_updates[key] = value.model_dump()

        self.preprocess_missing_vals_ui()
        self.save_prepared_data_ui()
        self.create_metadata()
        self.run_synthesis_ui()
        self.evaluate_ui()

        return self.synthetic_df


    # ============================================================================
    # Display previously completed results (UI)
    # ============================================================================

    # TODO: Improve how revised values are displayed.
    def display_revision_summary_ui(self) -> None:
        """Display previously selected revision changes (UI)."""
        updates = self.report.update.selected_plan_updates

        if not updates:
            return

        with st.container(border=True):
            st.subheader("Revision summary")

            if "update_mech" in updates:
                option = updates["update_mech"]
                st.write(f"**Generator:** {option['generator_name']}")
                st.write(f"**Reason:** {option['reason']}")

            if "update_params" in updates:
                st.write(f"**Updated parameters:** {updates['update_params']}")

            if "update_missing" in updates:
                option = updates["update_missing"]
                st.write(f"**Missing value handling:** {option['mode']}")
                st.write(f"**Reason:** {option['reason']}")


    def get_prev_evaluation_metrics(self):
        """Retrieve evaluation metrics from the previous workflow version (UI)."""
        if self.version_id == 1:
            return None, None

        prev_revision = st.session_state.revision_history.get(self.version_id - 1)

        if prev_revision is None:
            return None, None

        return prev_revision.report.evaluate.syndat, prev_revision.report.evaluate.attack


    def display_revision_results_ui(self) -> None:
        """Display previously generated and evaluated revision results (UI)."""

        # Synthetic dataset
        if self.synthetic_df is not None:
            st.subheader("Synthetic dataset")
            st.dataframe(self.synthetic_df.head())

        # Retrieve metrics from the previous version
        prev_syndat_metrics, prev_attack_metrics = self.get_prev_evaluation_metrics()

        # Evaluation metrics
        col1, col2 = st.columns(2)
        with col1:
            if self.report.evaluate.syndat is not None:
                display_metrics_ui(
                    metrics=self.report.evaluate.syndat, prev_metrics=prev_syndat_metrics,
                )

        with col2:
            if self.report.evaluate.attack is not None:
                display_attack_metrics_ui(
                    attack_metrics=self.report.evaluate.attack,
                    description=ANONYMETER_METRIC_DESCRIPTION,
                    key_prefix=f"review_v{self.version_id}",
                    prev_metrics=prev_attack_metrics,
                )

        # TODO: add some sort of metric summary.

        # TODO: add graph display after loading as well

        # Agent evaluation
        if self.agent_decision is not None:
            display_agent_evaluation_ui(self.agent_decision)


    # ============================================================================
    # Workflow execution
    # ============================================================================

    def run(self):
        """Run all synthetic data generation workflow steps."""
        self.load_dataframe()
        self.apply_preferences()

        # Preprocessing and generator selection
        self.preprocess_dtypes()
        self.generator_selection()
        self.preprocess_missing_vals()

        # Prepare for and run data synthesis tool
        self.save_prepared_data_cli()
        self.create_metadata()
        self.run_synthesis()

        # Evaluate results and make report
        self.evaluate()
        final_gen = self.post_evaluation_check()
        return final_gen.synthetic_df

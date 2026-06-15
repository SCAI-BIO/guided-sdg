import copy
from pathlib import Path
from typing import Any, Hashable
import os

import pandas as pd
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model, BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from agentic_synth_data.dataset_evaluation.evaluation import evaluation, write_report, calculate_metrics, evaluation_review
from agentic_synth_data.dataset_preparation.dtype_handling import handle_dtypes, missing_value_precheck
from agentic_synth_data.dataset_preparation.load_dataframe import load_df
from agentic_synth_data.dataset_preparation.semantic_type_handling import handle_semantic_types, prepare_for_syndat
from agentic_synth_data.dataset_preparation.metadata_formatting import format_json
from agentic_synth_data.dataset_preparation.missing_value_handling import handle_missing_values
from agentic_synth_data.preference_setup import select_preferences, SynthesisRequest, ImputationConfig
from agentic_synth_data.prompts.system_prompt import SYSTEM_PROMPT
from agentic_synth_data.rerun_workflow import RevisionOption, plan_update, rerun
from agentic_synth_data.synthetic_generation.generation import run_generator, select_generator, generator_profile
from agentic_synth_data.utils import format_response


class DataGen:
    def __init__(self, data_path):
        self.data_path:str = data_path
        
        self.version_id = 1
        self.parent_version = None

        self.synthesis_request:SynthesisRequest | None = None
        self.python_dtypes: dict[Hashable, str] | None = None
        self.semantic_types: dict[str, str] | None = None

        self.source_df: pd.DataFrame | None = None # Original data
        self.norm_na_df: pd.DataFrame | None = None # Original data (optional based on marker replacement)
        self.dtype_handled_df: pd.DataFrame | None = None # dtype prepared data
        self.missing_handled_df: pd.DataFrame | None = None # Missing value prepared data
        self.prepared_df: pd.DataFrame | None = None # Generator prepared data
        self.synthetic_df: pd.DataFrame | None = None # Synthetic data

        self.prepared_path: str | None = None # What is input into the tool (either with or without missing value handling)
        self.synthetic_path: str | None = None

        self.model:BaseChatModel = self.create_model()
        self.checkpointer = InMemorySaver()
        self.config:dict[str, dict[str, Any]] = {"configurable": {"thread_id": 1}}
        self.agent:CompiledStateGraph = self.create_agent()

        self.report: dict = {
            "input": {
                "path": self.data_path,
                "original_shape": {},
                "prepared_shape": {},
                "version_id": {},
                "parent_version": {},
            },
            "prep": {
                "missing_markers": {"detected": {}, "replaced": {},},
                "dtypes": {},
                "stypes": {},
                "missing_handling": {"before": {}, "after": {},},
            },
            "gen": {"prep_path": {}, "synth_path": {}, "synth_shape": {},},
            "review": {
                "jsd": {},
                "norm_corr_diff": {},
                "discrim_auc": {},
                "problem_columns": {},
                "recommendation": {},
                "rationale": {},
            },
        }


    def create_model(self) -> BaseChatModel:
        #print("="*25+"[Setting up model...]"+"="*25)
        model = init_chat_model(
            "MiniMaxAI/MiniMax-M2.5",
            model_provider="openai",
            base_url=os.environ.get("BASE_URL"),
            )
        #print("\n=> Model successfully created.")
        return model


    def create_agent(self):
        #print("\n\n")
        #print("="*25+"[Setting up agent...]"+"="*25)
        agent = create_agent(
            model=self.model,
            checkpointer=self.checkpointer,
            system_prompt=SYSTEM_PROMPT,
        )
        #print("=> Agent successfully created.")
        return agent


    def load_dataframe(self) -> pd.DataFrame:
        """Step 1: Setup source dataframe"""
        #print("\n\n")
        #print("="*25+"[STEP: Create the dataframe]"+"="*25)
        self.source_df = load_df(self.data_path)
        self.report["input"]["original_shape"] = self.source_df.shape
        self.report["input"]["version_id"] = self.version_id
        self.report["input"]["parent_version"] = self.parent_version
        return self.source_df


    def select_preferences(self):
        """Step 2: Setup preferences"""
        #print("\n\n")
        print("="*25+"[STEP: Select preferences]"+"="*25)
        self.synthesis_request = select_preferences()
        return self.synthesis_request


    def preprocess_dtypes(self) -> pd.DataFrame:
        """Step 3: Preprocess
            i. Check for missing value markers
            ii. Handle dtypes (both categories)
        """
        # Initial check for missing value markers (can affect dtypes)
        print("\n\n")
        print("="*25+"[STEP: Normalize missing values]"+"="*25)
        source_df = self.source_df.copy()
        norm_df, is_replaced, detected_markers, replaced_markers = missing_value_precheck(source_df)

        self.report["prep"]["missing_markers"]["detected"] = detected_markers
        self.report["prep"]["missing_markers"]["replaced"] = replaced_markers

        if is_replaced:
            self.norm_na_df = norm_df.copy()
            print("\n=> Dataframe has been saved for use during evaluation.")

        print("\n\n")
        print("="*25+"[STEP: Handle data types]"+"="*25)
        self.dtype_handled_df, self.python_dtypes = handle_dtypes(norm_df)

        self.report["prep"]["dtypes"] = self.python_dtypes

        print("\n\n")
        print("="*25+"[STEP: Handle semantic types]"+"="*25)
        self.semantic_types = handle_semantic_types(df=self.dtype_handled_df, agent=self.agent, config=self.config)

        self.report["prep"]["stypes"] = self.semantic_types
        return self.dtype_handled_df


    def generator_selection(self):
        """Step 3: Preprocess
            iii. Select/verify chosen generator
        """
        print("\n\n")
        print("="*25+"[STEP: Select data generator]"+"="*25)
        print()
        df = self.dtype_handled_df.copy()
        profile = generator_profile(df)
        self.synthesis_request = select_generator(synthesis_request=self.synthesis_request, semantic_types=self.semantic_types, profile=profile, agent=self.agent, config=self.config,)
        return self.synthesis_request


    def preprocess_missing_vals(self):
        """Step 3: Preprocess
            iv. Handle missing values (if needed for the generator)
        """
        self.report["prep"]["missing_handling"]["before"] = self.dtype_handled_df.isna().sum().sum()

        mechanism = self.synthesis_request.generator.name

        if mechanism in ["PATEGAN", "CTGAN",]:
            print("\n\n")
            print("="*25+"[STEP: Handle missing values]"+"="*25)

            self.missing_handled_df, self.synthesis_request = handle_missing_values(
                df=self.dtype_handled_df.copy(),
                synthesis_request=self.synthesis_request,
                semantic_types=self.semantic_types,
                agent=self.agent,
                config=self.config,
                )

            self.prepared_df = self.missing_handled_df.copy()
            self.report["input"]["prepared_shape"] = self.prepared_df.shape
            self.report["prep"]["missing_handling"]["after"] = self.missing_handled_df.isna().sum().sum()

            return self.prepared_df, self.synthesis_request

        else:
            self.prepared_df = self.dtype_handled_df.copy()
            self.report["input"]["prepared_shape"] = self.prepared_df.shape
            self.report["prep"]["missing_handling"]["after"] = self.prepared_df.isna().sum().sum()

            print(f"\nThe selected data generator does not require missing value handling: {mechanism}")

            # Set back to default
            if self.synthesis_request.missing_values.mode == "impute":
                self.synthesis_request.missing_values.impute = ImputationConfig()

            elif self.synthesis_request.missing_values.impute == "threshold":
                self.synthesis_request.missing_values.threshold = None

            self.synthesis_request.missing_values.mode = None

            return self.prepared_df, self.synthesis_request


    def save_prepared_data(self):
        """Step 4: Setup tool data
            i. Save the prepared data
        """
        print("\n\n")
        print("="*25+"[STEP: Create metadata for tool use]"+"="*25)
        input_path = Path(self.data_path)

        # The prepared data for generator tool input
        output_filename = f"{input_path.stem}_prepared.csv"
        output_dir = Path("../data")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.prepared_path = output_dir / output_filename
        self.prepared_df.to_csv(self.prepared_path, index=False)
        format_response(f"- Created csv for generator input: {self.prepared_path}")
        self.report["gen"]["prep_path"] = self.prepared_path
        return self.prepared_path


    def create_metadata(self):
        """Step 4: Setup tool data
            i. Create metadata for tool use
        """
        format_json(self.prepared_path, self.semantic_types)


    def run_synthesis(self):
        """Step 5: Synthesize data"""
        print("\n\n")
        print("="*25+"[STEP: Synthesize synthetic data]"+"="*25)

        self.synthetic_path = run_generator(
            data_path=self.data_path,
            clean_csv_path=self.prepared_path,
            synthesis_request=self.synthesis_request,
            version_id=self.version_id,
            )

        # The generators add an "Unnamed: 0" column, so check that all cols match:
        expected_columns = self.prepared_df.columns

        synth_df = pd.read_csv(self.synthetic_path, index_col=False)
        extra_columns = synth_df.columns.difference(expected_columns)

        if not extra_columns.empty:
            synth_df = synth_df.drop(columns=extra_columns)

        self.synthetic_df = synth_df

        self.report["gen"]["synth_path"] = self.synthetic_path
        self.report["gen"]["synth_shape"] = self.synthetic_df.shape
        return self.synthesis_request, self.synthetic_path, self.synthetic_df


    def evaluate(self):
        """Step 6: Evaluate results"""
        print("\n\n")
        print("="*25+"[STEP: Evaluate synthetic data]"+"="*25)
        synth_df = self.synthetic_df.copy()
        if self.norm_na_df is not None:
            real_df = self.norm_na_df.copy()
        else:
            real_df = self.source_df.copy()

        real_df, synth_df = prepare_for_syndat(real_df=real_df, synthetic_df=synth_df, semantic_types=self.semantic_types)

        metrics = calculate_metrics(real_df=real_df, synthetic_df=synth_df)
        self.report["review"]["jsd"] = metrics["jsd"]
        self.report["review"]["norm_corr_diff"] = metrics["norm_corr_diff"]
        self.report["review"]["discrim_auc"] = metrics["discrim_auc"]

        agent_decision = evaluation(
            synthesis_request=self.synthesis_request,
            metrics=metrics,
            agent=self.agent,
            config=self.config,
        )

        self.agent_decision = agent_decision

        self.report["review"]["problem_columns"] = agent_decision["problem_columns"]
        self.report["review"]["recommendation"] = agent_decision["recommendation"]
        self.report["review"]["rationale"] = agent_decision["rationale"]

        write_report(report=self.report, synthesis_request=self.synthesis_request, data_path=self.data_path, version_id=self.version_id)


    def post_evaluation_check(self):
        """Step 7: Check in with user and possibly"""
        print("\n\n")
        print("="*25+"[END: Evaluate workflow]"+"="*25)
        
        choice = evaluation_review(eval_decision=self.agent_decision, agent=self.agent, config=self.config)
    
        if choice == "revise":
            selected_updates =  plan_update(
                synthesis_request=self.synthesis_request,
                agent_decision=self.agent_decision,
                agent=self.agent,
                config=self.config,
            )

            if selected_updates is not None:
                self.report["review"]["selected_plan_updates"] = selected_updates.model_dump()
                
                revised = self.create_revision(selected_updates)
                return revised
            
            else:
                return self
        
        else:
            return self


    def run(self):
        self.load_dataframe()
        self.select_preferences()

        # Preprocessing and generator selection
        self.preprocess_dtypes()
        self.generator_selection()
        self.preprocess_missing_vals()

        # Prepare for and run data synthesis tool
        self.save_prepared_data()
        self.create_metadata()
        self.run_synthesis()

        # Evaluate results and make report
        self.evaluate()
        final_gen = self.post_evaluation_check()
        return final_gen.synthetic_df
    
    def create_revision(self, revision_choice:RevisionOption) -> "DataGen":
        """Copy datagen with the approved revisions. Currently can only change gnerator name"""
        print("\n" + "=" * 25 + "[REVISION: Create and update workflow copy]" + "=" * 25)
        revised = DataGen(self.data_path)

        revised.version_id = self.version_id + 1
        revised.parent_version = self.version_id

        revised.model = self.model
        revised.agent = self.agent
        revised.checkpointer = self.checkpointer
        revised.config = copy.deepcopy(self.config)

        revised.synthesis_request = self.synthesis_request.model_copy(deep=True)
        revised.synthesis_request.generator.name = revision_choice.generator_name
        revised.synthesis_request.generator.parameters = revision_choice.parameters

        revised.source_df = self.source_df.copy()
        revised.norm_na_df = self.norm_na_df.copy() if self.norm_na_df is not None else None
        revised.dtype_handled_df = self.dtype_handled_df.copy()
        revised.python_dtypes = copy.deepcopy(self.python_dtypes)
        revised.semantic_types = copy.deepcopy(self.semantic_types)

        revised.report = copy.deepcopy(self.report)
        revised.report["input"]["version_id"] = revised.version_id
        revised.report["input"]["parent_version"] = revised.parent_version

        revised._reset_from_generator_selection()
        revised._rerun_generator_selection(revision_choice=revision_choice)
        return revised


    def _rerun_generator_selection(self, revision_choice: RevisionOption) -> None:
        """Apply approved generator revision and rerun dependent workflow steps."""
        print("\n" + "=" * 25 + "[REVISION: Generator selection]" + "=" * 25)
        print(f"New generator: {revision_choice.generator_name}")
        print(f"Reason: {revision_choice.reason}")

        self.report["revisions"].append(
            {
                "rerun_step": revision_choice.rerun_step,
                "generator_name": revision_choice.generator_name,
                "parameters": revision_choice.parameters,
                "reason": revision_choice.reason,
            }
        )

        # Rerun steps from generator selection
        self.preprocess_missing_vals()
        self.save_prepared_data()
        self.create_metadata()
        self.run_synthesis()
        self.evaluate()


    def _reset_from_generator_selection(self) -> None:
        """Clear outputs that are expected to be updated after chaning the generator"""
        self.missing_handled_df = None
        self.prepared_df = None
        self.prepared_path = None
        self.metadata = None
        self.synthetic_df = None
        self.metrics = None
        self.agent_decision = None

        self.report["gen"] = {}
        self.report["review"] = {}

        self.report.setdefault("revisions", [])
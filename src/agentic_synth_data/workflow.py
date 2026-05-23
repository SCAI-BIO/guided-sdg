import os
import pandas as pd
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver

from agentic_synth_data.preference_setup import select_preferences
from agentic_synth_data.dataset_preparation.load_dataframe import load_df

from agentic_synth_data.dataset_preparation.dtype_handling import handle_dtypes, missing_value_precheck
from agentic_synth_data.dataset_preparation.semantic_type_handling import handle_semantic_types

from agentic_synth_data.dataset_preparation.missing_value_handling import handle_missing_values
from agentic_synth_data.dataset_preparation.metadata_formatting import format_json

from agentic_synth_data.synthetic_generation.generation import make_generator_tool, run_synthesis
from agentic_synth_data.dataset_evaluation.evaluation import evaluation

from agentic_synth_data.prompts.system_prompt import SYSTEM_PROMPT
# Note: usage_metadata could later be used to assess prompting and the workflow


class DataGen:
    def __init__(self, data_path):
        self.data_path:str = data_path
        self.synthesis_request:dict[str, Any] | None = None
        self.semantic_types: dict | None = None

        self.source_df: pd.DataFrame | None = None # Original data
        self.norm_na_df: pd.DataFrame | None = None # Only if markers were replaced
        self.dtype_handled_df: pd.DataFrame | None = None # dtype cleaned data
        self.missing_handled_df: pd.DataFrame | None = None # generator prepared data
        self.synthetic_df: pd.DataFrame | None = None # synthetic data

        self.clean_csv_path: str | None = None # more accurate term?
        self.synthetic_path: str | None = None

        self.model:BaseChatModel = self.create_model()
        self.checkpointer = InMemorySaver()
        self.config:dict = {"configurable": {"thread_id": 1}}

        self.generator_tool:BaseTool = self.create_generator_tool()
        self.agent:CompiledStateGraph = self.create_agent()


    def create_model(self) -> BaseChatModel:
        #print("="*25+"[Setting up model...]"+"="*25)
        model = init_chat_model(
            "MiniMaxAI/MiniMax-M2.5",
            model_provider="openai",
            base_url=os.environ.get("BASE_URL"),
            )
        #print("\n=> Model successfully created.")
        return model
    

    def create_generator_tool(self) -> BaseTool:
        return make_generator_tool(self)


    def create_agent(self):
        #print("\n\n")
        #print("="*25+"[Setting up agent...]"+"="*25)
        agent = create_agent(
            model=self.model,
            tools=[self.generator_tool],
            checkpointer=self.checkpointer,
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        self.generator_tool.name: {
                            "allowed_decisions": ["approve", "edit", "reject", "respond"]
                        }
                    },
                    description_prefix="Synthetic generator execution pending approval.",
                )
            ],
            system_prompt=SYSTEM_PROMPT,
        )
        #print("=> Agent successfully created.")
        return agent


    def select_preferences(self):
        #print("\n\n")
        print("="*25+"[STEP: Select preferences]"+"="*25)
        self.synthesis_request = select_preferences()
        return self.synthesis_request


    def load_dataframe(self) -> pd.DataFrame:
        #print("\n\n")
        #print("="*25+"[STEP: Create the dataframe]"+"="*25)
        self.source_df = load_df(self.data_path)#, self.synthesis_request["data_scope"]["selected_columns"])
        return self.source_df


    def preprocess(self) -> pd.DataFrame:
        # Initial check for missing value markers (can affect dtypes)
        print("\n\n")
        print("="*25+"[STEP: Normalize missing values]"+"="*25)
        norm_df, missing_markers_replaced = missing_value_precheck(self.source_df)

        if missing_markers_replaced:
            self.norm_na_df = norm_df.copy(deep=True)
            print("\n=> Dataframe has been saved for during evaluation.")

        print("\n\n")
        print("="*25+"[STEP: Handle data types]"+"="*25)
        # separate check for potential missing vlaues from handledtypes
        self.dtype_handled_df = handle_dtypes(norm_df)
        
        print("\n\n")
        print("="*25+"[STEP: Handle semantic types]"+"="*25)
        self.semantic_types = handle_semantic_types(df=self.dtype_handled_df, agent=self.agent, config=self.config)

        print("\n\n")
        print("="*25+"[STEP: Handle missing values]"+"="*25)
        self.missing_handled_df, self.synthesis_request = handle_missing_values(
            df=self.dtype_handled_df.copy(deep=True), 
            synthesis_request=self.synthesis_request, 
            semantic_types=self.semantic_types, 
            agent=self.agent, 
            config=self.config,
            )
        return self.missing_handled_df
    

    def save_clean_data(self):
        input_path = Path(self.data_path)

        output_filename = f"{input_path.stem}_cleaned.csv" # df prepared for generator tools

        output_dir = Path("../data")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.clean_csv_path = output_dir / output_filename

        self.missing_handled_df.to_csv(self.clean_csv_path, index=False)
        return self.clean_csv_path


    def create_metadata(self):
        print("\n\n")
        print("="*25+"[STEP: Create metadata for tool use]"+"="*25)
        format_json(self.clean_csv_path, self.semantic_types)


    def run_synthesis(self):
        print("\n\n")
        print("="*25+"[STEP: Plan and synthesize synthetic data]"+"="*25)
        self.synthesis_request, self.synthetic_path = run_synthesis(self)

        # The generators add an "Unnamed: 0" column, so check that all cols match:
        expected_columns = self.missing_handled_df.columns

        synth_df = pd.read_csv(self.synthetic_path, index_col=False)
        # TODO: test what happens if unnamed column is in the OG dataset.
        # What if the original column is Unnamed: 0?
        # if "Unnamed: 0" in self.synthetic_df.columns:
        #     self.synthetic_df = self.synthetic_df.drop(columns=["Unnamed: 0"])
        #print("initial", synth_df.head())
        extra_columns = synth_df.columns.difference(expected_columns)

        if not extra_columns.empty:
            synth_df = synth_df.drop(columns=extra_columns)

        self.synthetic_df = synth_df
        #print("after", self.synthetic_df.head())
        return self.synthesis_request, self.synthetic_path, self.synthetic_df

    
    def evaluate(self):
        print("\n\n")
        print("="*25+"[STEP: Evaluate synthetic data]"+"="*25)
        if self.norm_na_df is not None:
            real_df = self.norm_na_df.copy(deep=True)
        else:
            real_df = self.source_df.copy(deep=True)

        synth_df = self.synthetic_df.copy(deep=True)
        
        # Quick comparison look (probably move unnamed removal to generate)
        print(synth_df.head())
        synth_df = synth_df.loc[:, ~synth_df.columns.str.contains("^Unnamed")]
        print("\n updated", synth_df.head())

        self.evaluation_results = evaluation(
            real_df=real_df,
            synthetic_df=synth_df,
            synthesis_request=self.synthesis_request,
            agent=self.agent,
            config=self.config,
        )
        return self.evaluation_results


    def run(self):
        self.load_dataframe()
        self.select_preferences()
        self.preprocess()
        self.save_clean_data()
        self.create_metadata()
        self.run_synthesis()
        self.evaluate()
        return {
            "synthesis_request": self.synthesis_request,
            "preprocessed_df": self.missing_handled_df,
            "semantic_types": self.semantic_types,
            "clean_csv_path": self.clean_csv_path,
            "synthetic_path": self.synthetic_path,
            "synthetic_df": self.synthetic_df,
            "evaluation_results": self.evaluation_results,
        }
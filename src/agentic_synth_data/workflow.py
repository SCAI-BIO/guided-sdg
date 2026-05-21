import os
import pandas as pd
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from agentic_synth_data.preference_setup import select_preferences
from agentic_synth_data.dataset_preparation.load_dataframe import load_df

from agentic_synth_data.dataset_preparation.dtype_handling import handle_dtypes
from agentic_synth_data.dataset_preparation.semantic_type_handling import handle_semantic_types, match_to_semantic_types

from agentic_synth_data.dataset_preparation.missing_value_handling import handle_missing_values
from agentic_synth_data.dataset_preparation.metadata_formatting import format_json

from agentic_synth_data.synthetic_generation.generation import make_generator_tool, run_synthesis
from agentic_synth_data.dataset_evaluation.evaluation import evaluation

from agentic_synth_data.prompts.system_prompt import SYSTEM_PROMPT
# Note: usage_metadata could later be used to assess prompting and the workflow


class DataGen:
    def __init__(self, data_path):
        self.data_path = data_path
        self.synthesis_request = None
        self.df = None
        self.semantic_types = None

        self.clean_csv_path = None
        self.synthetic_path = None

        self.model = self.create_model()
        self.checkpointer = InMemorySaver()
        self.config = {"configurable": {"thread_id": 1}}

        self.generator_tool = self.create_generator_tool()
        self.agent = self.create_agent()


    def create_model(self):
        return init_chat_model(
            "MiniMaxAI/MiniMax-M2.5",
            model_provider="openai",
            base_url=os.environ.get("BASE_URL"),
            )
    

    def create_generator_tool(self):
        return make_generator_tool(self)

    def create_agent(self):
        return create_agent(
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


    def select_preferences(self):
        self.synthesis_request = select_preferences()
        return self.synthesis_request

    def load_and_preprocess(self):
        self.df = load_df(self.data_path)#, self.synthesis_request["data_scope"]["selected_columns"])

        self.df = handle_dtypes(self.df)
        self.df, self.semantic_types = handle_semantic_types(self.df)

        self.df, self.synthesis_request = handle_missing_values(
            self.df, 
            self.synthesis_request, 
            self.semantic_types, 
            agent=self.agent, 
            config=self.config,
            )
        return self.df
    
    def save_clean_data(self):
        input_path = Path(self.data_path)

        output_filename = f"{input_path.stem}_cleaned.csv"

        output_dir = Path("../data")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.clean_csv_path = output_dir / output_filename

        self.df.to_csv(self.clean_csv_path, index=False)
        return self.clean_csv_path


    def create_metadata(self):
        format_json(self.clean_csv_path, self.semantic_types)


    def run_synthesis(self):
       return run_synthesis(self)
    

    def evaluate(self):
        real_df = pd.read_csv(self.clean_csv_path)
        synth_df = pd.read_csv(self.synthetic_path)

        synth_df = synth_df.loc[:, ~synth_df.columns.str.contains("^Unnamed")]

        self.synthetic_df = synth_df

        self.evaluation_results = evaluation(
            real_df,
            synth_df,
            self.synthesis_request,
            agent=self.agent,
            config=self.config,
        )
        return self.evaluation_results

    def run(self):
        self.select_preferences()
        self.upload_and_preprocess()
        self.save_clean_data()
        self.create_metadata()
        self.run_synthesis()
        self.evaluate()
        return {
            "synthesis_request": self.synthesis_request,
            "cleaned_df": self.df,
            "semantic_types": self.semantic_types,
            "clean_csv_path": self.clean_csv_path,
            "synthetic_path": self.synthetic_path,
            "synthetic_df": self.synthetic_df,
            "evaluation_results": self.evaluation_results,
        }
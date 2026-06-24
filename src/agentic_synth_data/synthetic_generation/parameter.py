from typing import Any
from pydantic import BaseModel, Field


class PATEGANConfig(BaseModel):
    eps:int = Field()
    delta:float = 1e-5
    infer_ranges:bool = False
    num_teachers:int = 10
    n_inters:int = 100
    batch_size:int = 128
    learning_rate:float = 1e-4
    multiprocess:bool = False


class BayesianNetConfig(BaseModel):
    histogram_bins:int = 10
    degree:int = 1
    infer_ranges:bool = False
    multiprocess:bool = True
    seed:Any = None


class PrivBayesConfig(BaseModel):
    histogram_bins:int = 10
    degree:int = 1
    epsilon:float = .1
    infer_ranges:bool = False
    multiprocess:bool = True
    seed:Any = None


class CTGANConfig(BaseModel):
    embedding_dim:int = 128
    gen_dim:tuple[int, int] = (256, 256)
    dis_dim:tuple[int, int] = (256, 256)
    l2scale:float = 1e-6
    batch_size:int = 500
    epochs:int = 300
    multiprocess:bool = False


CONFIG_MAP = {
   "PATEGAN": PATEGANConfig,
   "BayesianNet": BayesianNetConfig,
   "PrivBayes": PrivBayesConfig,
   "CTGAN": CTGANConfig,
}


def format_parameters(parameter_output:list, ):
    """Format expected colon separated string"""
    parameters = ":".join(parameter_output)
    return parameters
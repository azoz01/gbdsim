from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from torch import nn


class MomentumConfig(BaseModel):
    type: Literal["momentum"]
    hidden_size: int
    n_hidden: int
    output_size: int
    activation_function: nn.Module

    @model_validator(mode="before")
    def parse_validation_function(cls, data) -> dict[str, Any]:
        data = deepcopy(data)
        preprocess_activation_function(data)
        return data

    class Config:
        arbitrary_types_allowed = True


class Col2NodeConfig(BaseModel):
    type: Literal["col2node"]
    f_n_hidden: int
    f_hidden_size: int
    g_n_hidden: int
    g_hidden_size: int
    output_size: int
    activation_function: nn.Module

    @model_validator(mode="before")
    def parse_validation_function(cls, data) -> dict[str, Any]:
        data = deepcopy(data)
        preprocess_activation_function(data)
        return data

    class Config:
        arbitrary_types_allowed = True


def preprocess_activation_function(data: dict[str, Any]) -> None:
    assert (
        "activation_function" in data
    ), "Data does not have activation_function"
    assert hasattr(
        nn, data["activation_function"]
    ), "activation_function is not a valid activation function"
    data["activation_function"] = getattr(nn, data["activation_function"])()


class GraphSageConfig(BaseModel):
    hidden_channels: int
    num_layers: int
    out_channels: int


class GBDSimConfig(BaseModel):
    type: Literal["gbdsim"]
    col2node_config: Col2NodeConfig | MomentumConfig = Field(
        discriminator="type"
    )
    graph_sage_config: GraphSageConfig
    similarity_head_strategy: Literal["euclidean", "nn"] = "euclidean"


class Dataset2VecConfig(BaseModel):
    type: Literal["dataset2vec"]


class DataConfig(BaseModel):
    train_dataset_size: int
    train_batch_size: int
    val_dataset_size: int
    val_batch_size: int


class TrainingConfig(BaseModel):
    num_epochs: int
    learning_rate: float
    weight_decay: float


class ExperimentConfig(BaseModel):
    model: GBDSimConfig | Dataset2VecConfig = Field(discriminator="type")
    data: DataConfig
    training: TrainingConfig

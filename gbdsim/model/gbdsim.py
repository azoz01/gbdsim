from __future__ import annotations

from typing import Literal, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch_geometric.nn.models import GraphSAGE

from ..experiment_config import GBDSimConfig
from ..utils.constants import DEVICE
from ..utils.modules import (
    EuclideanDistance,
    MultiInputSequential,
    SimilarityInputProcessor,
)
from .column2node import Column2NodeLayer, MomentumLayer


class GBDSim(pl.LightningModule):

    def __init__(
        self,
        col2node: Column2NodeLayer | MomentumLayer = Column2NodeLayer(),
        graph_sage_hidden_channels=64,
        graph_sage_num_layers=3,
        graph_sage_out_channels=64,
        similarity_head_strategy: Literal["euclidean", "nn"] = "euclidean",
    ):
        super().__init__()
        self.col2node = col2node.to(DEVICE)
        self.output_dim = graph_sage_out_channels
        self.representation_layer = nn.Sequential(
            nn.Linear(graph_sage_out_channels, graph_sage_out_channels),
            col2node.activation_function,
            nn.Linear(graph_sage_out_channels, graph_sage_out_channels),
            col2node.activation_function,
            nn.Linear(graph_sage_out_channels, graph_sage_out_channels),
            col2node.activation_function,
        )
        self.edge_generation_network = nn.Sequential(
            nn.Linear(self.col2node.output_size, self.col2node.output_size),
            nn.LeakyReLU(),
            nn.Linear(
                self.col2node.output_size, self.col2node.output_size * 2
            ),
            nn.LeakyReLU(),
            nn.Linear(
                self.col2node.output_size * 2, self.col2node.output_size * 2
            ),
            nn.LeakyReLU(),
            nn.Linear(
                self.col2node.output_size * 2, self.col2node.output_size
            ),
            nn.LeakyReLU(),
        )
        self.edge_classification_network = nn.Sequential(
            nn.Linear(
                self.col2node.output_size, self.col2node.output_size * 2
            ),
            nn.LeakyReLU(),
            nn.Linear(
                self.col2node.output_size * 2, self.col2node.output_size * 2
            ),
            nn.LeakyReLU(),
            nn.Linear(
                self.col2node.output_size * 2, self.col2node.output_size * 2
            ),
            nn.LeakyReLU(),
            nn.Linear(self.col2node.output_size * 2, 1),
            nn.Sigmoid(),
        )
        self.gnn = GraphSAGE(
            -1,
            graph_sage_hidden_channels,
            graph_sage_num_layers,
            graph_sage_out_channels,
            0.0,
            self.col2node.activation_function,
        )

        if similarity_head_strategy == "euclidean":
            self.similarity_layer = EuclideanDistance()
        elif similarity_head_strategy == "nn":
            self.similarity_layer = MultiInputSequential(
                SimilarityInputProcessor(),
                nn.Linear(
                    3 * graph_sage_out_channels, graph_sage_out_channels
                ),
                nn.LeakyReLU(),
                nn.Linear(graph_sage_out_channels, 1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(
                f"Unknown similarity head strategy: {similarity_head_strategy}"
            )

    def forward(
        self,
        X1: torch.Tensor,
        y1: torch.Tensor,
        X2: torch.Tensor,
        y2: torch.Tensor,
    ) -> torch.Tensor:
        return self.calculate_dataset_distance(X1, y1, X2, y2)

    def calculate_dataset_origin_probability(
        self,
        X1: torch.Tensor,
        y1: torch.Tensor,
        X2: torch.Tensor,
        y2: torch.Tensor,
    ) -> torch.Tensor:
        dist = self.calculate_dataset_distance(X1, y1, X2, y2)
        return torch.clamp(torch.exp(-dist), 1e-10, 1 - 1e-10)

    def calculate_dataset_distance(
        self,
        X1: torch.Tensor,
        y1: torch.Tensor,
        X2: torch.Tensor,
        y2: torch.Tensor,
    ) -> torch.Tensor:
        enc1 = self.calculate_dataset_representation(X1, y1)
        enc2 = self.calculate_dataset_representation(X2, y2)
        return self.similarity_layer(enc1, enc2)

    def calculate_dataset_representation(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        g = self.convert_dataset_to_graph(X, y)
        return self.representation_layer(self.gnn(*g).mean(dim=0)).unsqueeze(0)

    def convert_dataset_to_graph(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes_features = self.col2node(X, y)
        edge_index = (
            self.__generate_pairs(torch.arange(X.shape[1]))
            .reshape(-1, 2)
            .to(DEVICE)
        )
        nodes_pairs = self.__generate_pairs(nodes_features)
        nodes_pairs = nodes_pairs[edge_index[:, 0] != edge_index[:, 1]]
        edge_relevance = (
            nn.ReLU()(
                self.edge_classification_network(nodes_pairs.sum(dim=1)) - 0.5
            )
            / 0.5
        )
        connectivity = (
            self.edge_generation_network(nodes_pairs.sum(dim=1))
            * edge_relevance
        )
        edge_index = edge_index[edge_index[:, 0] != edge_index[:, 1]].reshape(
            2, -1
        )

        return (
            nodes_features,
            edge_index,
            connectivity,
        )

    def __generate_pairs(self, t: torch.Tensor) -> torch.Tensor:
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        return torch.stack(
            [
                t.repeat(t.shape[0], 1),
                torch.repeat_interleave(t, t.shape[0], 0),
            ],
            dim=1,
        ).reshape(t.shape[0] ** 2, 2, -1)

    @classmethod
    def from_config(cls, config: GBDSimConfig) -> GBDSim:
        col2node_config = config.col2node_config
        col2node_type = col2node_config.type
        col2node_kwargs = col2node_config.model_dump(exclude={"type"})
        if col2node_type == "col2node":
            col2node = Column2NodeLayer(**col2node_kwargs)
        else:
            col2node = MomentumLayer(**col2node_kwargs)
        return cls(
            col2node,
            graph_sage_hidden_channels=config.graph_sage_config.hidden_channels,  # noqa: E501
            graph_sage_num_layers=config.graph_sage_config.num_layers,
            graph_sage_out_channels=config.graph_sage_config.out_channels,
            similarity_head_strategy=config.similarity_head_strategy,
        )

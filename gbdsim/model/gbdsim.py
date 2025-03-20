from __future__ import annotations

from itertools import product
from typing import Tuple

import pytorch_lightning as pl
from torch import Tensor, bool, empty, exp, eye, int64, nn, sqrt
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import TransformerConv

from ..experiment_config import GBDSimConfig
from ..utils.constants import DEVICE
from .column2node import Column2NodeLayer, MomentumLayer
from .connectivity import pairwise_mutual_information


class GBDSim(pl.LightningModule):

    def __init__(
        self,
        col2node: Column2NodeLayer | MomentumLayer = Column2NodeLayer(),
        graph_sage_hidden_channels=64,
        graph_sage_num_layers=3,
        graph_sage_out_channels=64,
    ):
        super().__init__()
        self.col2node = col2node.to(DEVICE)
        self.output_dim = graph_sage_out_channels // 2
        self.representation_layer = nn.Sequential(
            nn.Linear(graph_sage_out_channels, graph_sage_out_channels // 2),
            col2node.activation_function,
        )
        self.gnn = Sequential(
            "x, edge_index, edge_features",
            [
                (
                    TransformerConv(
                        -1, out_channels=graph_sage_out_channels, edge_dim=1
                    ).to(DEVICE),
                    "x, edge_index, edge_features -> x",
                ),
                col2node.activation_function,
                (
                    TransformerConv(
                        -1, out_channels=graph_sage_out_channels, edge_dim=1
                    ).to(DEVICE),
                    "x, edge_index, edge_features -> x",
                ),
                col2node.activation_function,
            ],
        )
        self.similarity_layer = nn.Sequential(
            nn.Linear(3 * graph_sage_out_channels, 1),
            nn.ReLU(),
        )
        self.save_hyperparameters()

    def forward(
        self, X1: Tensor, y1: Tensor, X2: Tensor, y2: Tensor
    ) -> Tensor:
        return self.calculate_dataset_distance(X1, y1, X2, y2)

    def calculate_dataset_origin_probability(
        self,
        X1: Tensor,
        y1: Tensor,
        X2: Tensor,
        y2: Tensor,
    ) -> Tensor:
        dist = self.calculate_dataset_distance(X1, y1, X2, y2)
        return exp(-dist)

    def calculate_dataset_distance(
        self, X1: Tensor, y1: Tensor, X2: Tensor, y2: Tensor
    ) -> Tensor:
        enc1 = self.calculate_dataset_representation(X1, y1)
        enc2 = self.calculate_dataset_representation(X2, y2)
        # concatenated = concat([enc1, enc2, enc1 * enc2], dim=1)
        # return self.similarity_layer(concatenated)
        return sqrt(((enc1 - enc2) ** 2).sum())

    def calculate_dataset_representation(self, X: Tensor, y: Tensor) -> Tensor:
        g = self.convert_dataset_to_graph(X, y)
        return self.representation_layer(self.gnn(*g).mean(dim=0)).unsqueeze(0)

    def convert_dataset_to_graph(
        self, X: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        nodes_features = self.col2node(X, y)
        mi = pairwise_mutual_information(X)
        connectivity = mi / mi.sum()
        connectivity = connectivity[
            ~eye(connectivity.shape[0], dtype=bool, device=connectivity.device)
        ]
        connectivity = connectivity.reshape(-1, 1).to(DEVICE)
        if X.shape[1] == 1:
            edge_index = empty(2, 0).type(int64).to(DEVICE)
        else:
            edge_index = (
                Tensor(
                    [
                        [i, j]
                        for i, j in product(
                            range(X.shape[1]), range(X.shape[1])
                        )
                        if i != j
                    ]
                )
                .type(int64)
                .permute(1, 0)
                .to(DEVICE)
            )
        return (
            nodes_features,
            edge_index,
            connectivity,
        )

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
        )

from __future__ import annotations

from itertools import product
from typing import Tuple

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor, bool, eye, int64
from torch_geometric.nn.models import GraphSAGE

from ..utils.constants import DEVICE
from .column2node import Column2NodeLayer, MomentumLayer
from .config import GbdsimConfig
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
        self.gnn = GraphSAGE(
            -1,
            graph_sage_hidden_channels,
            graph_sage_num_layers,
            graph_sage_out_channels,
        ).to(DEVICE)

    def forward(
        self, X1: Tensor, y1: Tensor, X2: Tensor, y2: Tensor
    ) -> Tensor:
        g1 = self.convert_dataset_to_graph(X1, y1)
        enc1 = self.gnn(*g1).mean(dim=0).unsqueeze(0)
        g2 = self.convert_dataset_to_graph(X2, y2)
        enc2 = self.gnn(*g2).mean(dim=0).unsqueeze(0)
        return F.cosine_similarity(enc1, enc2)

    def convert_dataset_to_graph(
        self, X: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        nodes_features = self.col2node(X, y)
        mi = pairwise_mutual_information(X)
        connectivity = mi / mi.sum()
        connectivity = connectivity[
            ~eye(connectivity.shape[0], dtype=bool, device=connectivity.device)
        ]
        edge_index = (
            Tensor(
                [
                    [i, j]
                    for i, j in product(range(X.shape[1]), range(X.shape[1]))
                    if i != j
                ]
            )
            .type(int64)
            .T.to(DEVICE)
        )
        return nodes_features, edge_index, connectivity

    @staticmethod
    def from_config(config: GbdsimConfig) -> GBDSim:
        col2node_config = config.col2node_config
        col2node_type = col2node_config.col2node_type
        col2node_kwargs = col2node_config.model_dump(exclude={"col2node_type"})
        if col2node_type == "col2node":
            col2node = Column2NodeLayer(**col2node_kwargs)
        else:
            col2node = MomentumLayer(**col2node_kwargs)
        return GBDSim(
            col2node,
            graph_sage_hidden_channels=config.graph_sage_config.hidden_channels,  # noqa: E501
            graph_sage_num_layers=config.graph_sage_config.num_layers,
            graph_sage_out_channels=config.graph_sage_config.out_channels,
        )

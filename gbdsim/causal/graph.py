import random
from itertools import product
from typing import Tuple

import networkx as nx
import torch

from .node import CausalNode


class CausalGraph:

    def __init__(
        self, nodes: list[list[CausalNode]], weights: list[torch.Tensor]
    ):
        self.nodes = nodes
        self.weights = weights

    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.generate_causal_matrix()
        sample_mask = torch.rand(size=[data.shape[0]]) <= 0.5
        if (
            sample_mask.sum() < 2
        ):  # edge case when not enough nodes is selected
            sample_mask[0] = True
            sample_mask[-1] = True
        selected_data = data[sample_mask].T
        selected_data = selected_data[
            :, torch.randperm(selected_data.shape[1])
        ]
        X, y = selected_data[:, :-1], selected_data[:, -1]
        y_pivot = random.choice(y)
        y = (y >= y_pivot).type(torch.int)
        return X, y

    def generate_causal_matrix(self) -> torch.Tensor:
        n_rows = random.randint(100, 10000)
        data = [
            torch.stack(
                [
                    node.noise_distribution.sample((n_rows,))
                    for node in self.nodes[0]
                ]
            )
        ]
        for i, (nodes, layer) in enumerate(zip(self.nodes[1:], self.weights)):
            data.append(layer @ data[-1])
            for i, node in enumerate(nodes):
                data[-1][i] = node.activation_function(
                    data[-1][i]
                ) + node.noise_distribution.sample((n_rows,))
        data = torch.concat(data, dim=0)
        return data

    @property
    def nx_graph(self) -> nx.Graph:
        G = nx.DiGraph()
        for node in self.nodes[0]:
            G.add_node(
                node,
                activation_function=node.activation_function,
                noise_std=node.noise_std,
            )
        G.add_nodes_from(self.nodes[0])
        nodes_idx_offset = len(self.nodes[0])
        last_layer_size = len(self.nodes[0])
        for nodes, weights in zip(self.nodes[1:], self.weights):
            weights = weights.to("cpu")
            for node in nodes:
                G.add_node(
                    node,
                    activation_function=node.activation_function,
                    noise_std=node.noise_std,
                )
            for i, j in product(
                range(weights.shape[0]), range(weights.shape[1])
            ):
                if torch.abs(weights[i, j]) > 1e-10:
                    u_idx = nodes_idx_offset - last_layer_size + j
                    v_idx = nodes_idx_offset + i
                    G.add_edge(u_idx, v_idx, weight=weights[i, j])
            nodes_idx_offset += len(nodes)
            last_layer_size = len(nodes)
        return G

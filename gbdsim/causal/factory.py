import random

import torch

from ..utils.constants import ACTIVATION_FUNCTIONS
from ..utils.distributions import generate_tnlu
from .graph import CausalGraph
from .node import CausalNode


class MlpCausalGraphFractory:
    @staticmethod
    def generate_causal_graph() -> CausalGraph:
        # Generate causal model parameters
        dropout_rate = (
            0.6
            * torch.distributions.beta.Beta(
                *torch.distributions.uniform.Uniform(0.1, 5.0).sample([2])
            )
            .sample()
            .item()
        )
        n_layers = int(generate_tnlu(1, 6, 3, 6))
        n_nodes_per_layer = int(generate_tnlu(4, 6, 3, 7))
        mlp_weights_std = generate_tnlu(
            0.01, 10.0, 0.001, 100, round_output=False
        )
        nodes_at_first_layer = int(generate_tnlu(1, 6, 3, 7))

        # Generates nodes list (each element of list
        # is list of nodes in specific layer)
        nodes = [
            MlpCausalGraphFractory.__generate_layer(nodes_at_first_layer)
        ] + [
            MlpCausalGraphFractory.__generate_layer(n_nodes_per_layer)
            for _ in range(n_layers - 1)
        ]

        # Generate edges weights
        # Rows are target nodes and columns are source nodes
        weights = torch.distributions.normal.Normal(0, mlp_weights_std).sample(
            [
                n_nodes_per_layer,
                n_nodes_per_layer * (n_layers - 1) + nodes_at_first_layer,
            ]
        )
        # Dropout edges
        masked_weights = weights * (
            torch.rand(size=weights.shape) >= dropout_rate
        ).type(torch.int32)
        # Convert weights to list of weights in specific layer
        # Note that len(edges) == len(nodes) - 1
        final_weights = [masked_weights[:, :nodes_at_first_layer]] + [
            masked_weights[
                :,
                nodes_at_first_layer
                + i * n_nodes_per_layer : nodes_at_first_layer  # noqa: E203
                + (i + 1) * n_nodes_per_layer,
            ]
            for i in range(n_layers - 2)
        ]
        return CausalGraph(nodes, final_weights)

    @staticmethod
    def __generate_layer(size: int) -> list[CausalNode]:
        return [
            CausalNode(
                random.choice(ACTIVATION_FUNCTIONS),
                generate_tnlu(0.1, 0.3, 0.001, 10, round_output=False),
            )
            for _ in range(size)
        ]

import networkx as nx
from torch import Tensor, nn

from gbdsim.causal.graph import CausalGraph
from gbdsim.causal.node import CausalNode


def sample_graph_nodes() -> list[CausalNode]:
    return [
        CausalNode(nn.Identity(), 1),
        CausalNode(nn.Identity(), 1),
        CausalNode(nn.ReLU(), 2),
        CausalNode(nn.ReLU(), 2),
        CausalNode(nn.ReLU(), 2),
        CausalNode(nn.ReLU(), 2),
        CausalNode(nn.Tanh(), 3),
        CausalNode(nn.Tanh(), 3),
        CausalNode(nn.Tanh(), 3),
    ]


def sample_edges_weights_matrix() -> list[Tensor]:
    return [
        Tensor([[1, 0], [2, 3], [0, 0], [0, 4]]),
        Tensor([[5, 0, 0, 0], [0, 6, 0, 8], [0, 0, 7, 0]]),
    ]


def sample_graph_edges() -> list[list[int]]:
    return [
        [0, 2, 1],
        [0, 3, 2],
        [1, 3, 3],
        [1, 5, 4],
        [2, 6, 5],
        [3, 7, 6],
        [5, 7, 8],
        [4, 8, 7],
    ]


def test_causal_graph_generates_proper_nx_representation() -> None:
    # Given
    input_nodes = sample_graph_nodes()
    G = CausalGraph(
        nodes=[input_nodes[:2], input_nodes[2:6], input_nodes[6:]],
        weights=sample_edges_weights_matrix(),
    )

    expected_nx_graph = nx.DiGraph()
    for node in input_nodes:
        expected_nx_graph.add_node(
            node,
            activation_function=node.activation_function,
            noise_std=node.noise_std,
        )
    for u, v, weight in sample_graph_edges():
        expected_nx_graph.add_edge(u, v, weight=weight)

    # When
    actual_nx_graph = G.nx_graph

    # Then
    assert expected_nx_graph.edges() == actual_nx_graph.edges()
    assert expected_nx_graph.nodes() == actual_nx_graph.nodes()

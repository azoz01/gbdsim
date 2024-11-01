from functools import reduce
from operator import add

import networkx as nx
from pytest import fixture
from torch import Tensor, nn

from gbdsim.causal.graph import CausalGraph
from gbdsim.causal.node import CausalNode
from gbdsim.utils.constants import DEVICE


@fixture
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


@fixture
def sample_graph_nodes_deterministic() -> list[CausalNode]:
    return [
        CausalNode(nn.Identity(), 1),
        CausalNode(nn.Identity(), 1),
        CausalNode(nn.Identity(), 1e-10),
        CausalNode(nn.Identity(), 1e-10),
        CausalNode(nn.Identity(), 1e-10),
        CausalNode(nn.Identity(), 1e-10),
        CausalNode(nn.Identity(), 1e-10),
        CausalNode(nn.Identity(), 1e-10),
        CausalNode(nn.Identity(), 1e-10),
    ]


@fixture
def sample_edges_weights_matrix() -> list[Tensor]:
    return [
        Tensor([[1, 0], [2, 3], [0, 0], [0, 4]]).to(DEVICE),
        Tensor([[5, 0, 0, 0], [0, 6, 0, 8], [0, 0, 7, 0]]).to(DEVICE),
    ]


@fixture
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


def test_causal_graph_generates_proper_nx_representation(
    sample_graph_nodes: list[CausalNode],
    sample_edges_weights_matrix: list[Tensor],
    sample_graph_edges: list[list[int]],
) -> None:
    # Given
    input_nodes = sample_graph_nodes
    G = CausalGraph(
        nodes=[input_nodes[:2], input_nodes[2:6], input_nodes[6:]],
        weights=sample_edges_weights_matrix,
    )

    expected_nx_graph = nx.DiGraph()
    for node in input_nodes:
        expected_nx_graph.add_node(
            node,
            activation_function=node.activation_function,
            noise_std=node.noise_std,
        )
    for u, v, weight in sample_graph_edges:
        expected_nx_graph.add_edge(u, v, weight=weight)

    # When
    actual_nx_graph = G.nx_graph

    # Then
    assert expected_nx_graph.edges() == actual_nx_graph.edges()
    assert expected_nx_graph.nodes() == actual_nx_graph.nodes()


def test_causal_graph_generates_data_with_proper_dependencies(
    sample_graph_nodes_deterministic: list[CausalNode],
    sample_edges_weights_matrix: list[Tensor],
    sample_graph_edges: list[list[int]],
) -> None:
    # Given
    input_nodes = sample_graph_nodes_deterministic
    G = CausalGraph(
        nodes=[input_nodes[:2], input_nodes[2:6], input_nodes[6:]],
        weights=sample_edges_weights_matrix,
    )
    edges_to_verify = [
        [0, 2, 1],
        [0, 3, 2],
        [1, 3, 3],
        [1, 5, 4],
        [2, 6, 5],
        [3, 7, 6],
        [5, 7, 8],
        [4, 8, 7],
    ]

    # When
    sample_data = G.generate_causal_matrix()

    # Then
    for i in range(9):
        parents = list(filter(lambda e: e[1] == i, edges_to_verify))
        if len(parents) > 0:
            actual_values = sample_data[i]
            expected_values = reduce(
                add,
                [
                    sample_data[parent_idx] * weight
                    for parent_idx, _, weight in parents
                ],
            )
            assert ((actual_values - expected_values).abs() <= 1e-5).all()


def test_causal_graph_generates_data_with_proper_dimensionality(
    sample_graph_nodes_deterministic: list[CausalNode],
    sample_edges_weights_matrix: list[Tensor],
) -> None:
    # Given
    input_nodes = sample_graph_nodes_deterministic
    G = CausalGraph(
        nodes=[input_nodes[:2], input_nodes[2:6], input_nodes[6:]],
        weights=sample_edges_weights_matrix,
    )

    # When
    X, y = G.generate_data()

    # Then
    assert X.shape[0] == y.shape[0]

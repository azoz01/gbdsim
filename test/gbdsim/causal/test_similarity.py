from torch import nn

from gbdsim.causal.similarity import (
    edge_del_cost,
    edge_ins_cost,
    edge_subst_cost,
    node_del_cost,
    node_ins_cost,
    node_match,
    node_subst_cost,
)


def test_node_match() -> None:
    assert not node_match(
        {"activation_function": nn.Identity()},
        {"activation_function": nn.ReLU()},
    )
    assert node_match(
        {"activation_function": nn.Identity()},
        {"activation_function": nn.Identity()},
    )


def test_node_match_when_any_node_empty() -> None:
    assert not node_match({}, {"activation_function": nn.Identity()})
    assert not node_match({"activation_function": nn.Identity()}, {})
    assert node_match({}, {})


def test_node_subst_cost() -> None:
    assert (
        node_subst_cost(
            {"activation_function": nn.ReLU(), "noise_std": 3.5},
            {"activation_function": nn.Identity(), "noise_std": 0.5},
        )
        == 4
    )


def test_node_subst_cost_when_any_node_empty() -> None:
    assert (
        node_subst_cost(
            {}, {"activation_function": nn.Identity(), "noise_std": 0.5}
        )
        == 1.5
    )
    assert (
        node_subst_cost(
            {"activation_function": nn.Identity(), "noise_std": 0.5}, {}
        )
        == 1.5
    )
    assert node_subst_cost({}, {}) == 0


def test_node_del_cost() -> None:
    assert (
        node_del_cost({"activation_function": nn.Identity(), "noise_std": 0.5})
        == 1.5
    )


def test_node_del_cost_when_empty() -> None:
    assert node_del_cost({}) == 0


def test_node_ins_costt() -> None:
    assert (
        node_ins_cost({"activation_function": nn.Identity(), "noise_std": 0.5})
        == 1.5
    )


def test_node_ins_cost_when_empty() -> None:
    assert node_ins_cost({}) == 0


def test_edge_subst_cost() -> None:
    assert edge_subst_cost({"weight": 0.4}, {"weight": 0.9}) == 0.5


def test_edge_del_cost() -> None:
    assert edge_del_cost({"weight": 0.4}) == 0.4


def test_edge_ins_cost() -> None:
    assert edge_ins_cost({"weight": 0.4}) == 0.4

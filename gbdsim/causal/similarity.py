from typing import Any, Mapping

import networkx as nx


def node_match(n1: Mapping[str, Any], n2: Mapping[str, Any]) -> bool:
    if len(n1) == 0 and len(n2) == 0:
        return True
    if len(n1) + len(n2) > 0:
        return False
    return n1["activation_function"] == n2["activation_function"]


def node_subst_cost(n1: Mapping[str, Any], n2: Mapping[str, Any]) -> float:
    if len(n1) == 0 and len(n2) == 0:
        return 0
    if len(n1) > 0 and len(n2) == 0:
        return node_ins_cost(n1)
    if len(n2) > 0 and len(n1) == 0:
        return node_ins_cost(n2)
    return (n1["activation_function"] == n2["activation_function"]) + abs(
        n1["noise_std"] - n2["noise_std"]
    )


def node_del_cost(n: Mapping[str, Any]) -> float:
    if len(n) == 0:
        return 0
    return 1 + n["noise_std"]


def node_ins_cost(n: Mapping[str, Any]) -> float:
    if len(n) == 0:
        return 0
    return 1 + n["noise_std"]


def edge_subst_cost(e1: Mapping[str, Any], e2: Mapping[str, Any]) -> float:
    return abs(e1["weight"] - e2["weight"])


def edge_del_cost(e: Mapping[str, Any]) -> float:
    return e["weight"]


def edge_ins_cost(e: Mapping[str, Any]) -> float:
    return e["weight"]


def calculate_graph_similarity(g1: nx.Graph, g2: nx.Graph) -> float:
    gen = nx.optimize_graph_edit_distance(
        g1,
        g2,
        node_match=node_match,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=edge_del_cost,
        edge_ins_cost=edge_ins_cost,
    )
    return next(gen)
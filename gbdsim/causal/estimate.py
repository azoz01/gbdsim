from itertools import chain, product
from operator import itemgetter
from typing import Tuple

import networkx as nx
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


def test_variables_dependence(
    feature_1: pd.Series, feature_2: pd.Series
) -> bool:
    _, p_value = spearmanr(feature_1, feature_2)
    return p_value.item() < 0.05  # type: ignore[no-untyped-call]


def compute_edge_weights_for_nodes(
    df: pd.DataFrame, all_features: list[Tuple[str, str]], target_feature
) -> list[Tuple[str, str, float]]:
    input_features = list(
        map(
            itemgetter(0),
            filter(lambda t: t[1] == target_feature, all_features),
        )
    )
    X = df[input_features]
    y = df[target_feature]
    coefficients = LinearRegression().fit(X, y).coef_
    return [
        (input_features[i], target_feature, coefficients[i].item())
        for i in range(len(input_features))
    ]


def estimate_causal_graph(df: pd.DataFrame) -> nx.DiGraph:
    feature_pairs = list(
        filter(lambda t: t[0] != t[1], product(df.columns, df.columns))
    )
    significant_pairs = []
    for feature_1, feature_2 in feature_pairs:
        if test_variables_dependence(df[feature_1], df[feature_2]):
            significant_pairs.append((feature_1, feature_2))
    features_to_nodes_mapping = dict(
        [(faeture, i) for i, faeture in enumerate(df.columns)]
    )
    edges = list(
        chain(
            *[
                compute_edge_weights_for_nodes(df, significant_pairs, feature)
                for feature in df.columns
            ]
        )
    )
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(df.columns))))
    G.add_edges_from(
        [
            (
                features_to_nodes_mapping[edge[0]],
                features_to_nodes_mapping[edge[1]],
                {"weight": edge[2]},
            )
            for edge in edges
        ]
    )
    return G

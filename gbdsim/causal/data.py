from typing import Tuple

from torch import Tensor

from .factory import MlpCausalGraphFractory
from .similarity import calculate_graph_distance


def generate_synthetic_causal_data_example() -> (
    Tuple[Tensor, Tensor, Tensor, Tensor, float]
):
    g1 = MlpCausalGraphFractory.generate_causal_graph()
    g2 = MlpCausalGraphFractory.generate_causal_graph()
    return (
        *g1.generate_data(),
        *g2.generate_data(),
        calculate_graph_distance(g1.nx_graph, g2.nx_graph),
    )

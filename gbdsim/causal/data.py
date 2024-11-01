from .factory import MlpCausalGraphFractory
from .similarity import calculate_graph_similarity


def generate_synthetic_causal_data_example():
    g1 = MlpCausalGraphFractory.generate_causal_graph()
    g2 = MlpCausalGraphFractory.generate_causal_graph()
    return (
        g1.generate_data(),
        g2.generate_data(),
        calculate_graph_similarity(g1.nx_graph, g2.nx_graph),
    )

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from gbdsim.causal.data import (
    MlpCausalGraphFractory,
    generate_synthetic_causal_data_example,
)
from gbdsim.causal.similarity import calculate_graph_distance
from gbdsim.model.gbdsim import GBDSim
from gbdsim.utils.constants import DEVICE


class MetricLearningResults:

    def evaluate_model(
        self,
        model: GBDSim,
        pairwise_similarity_eval_num_examples: int,
        retrieval_eval_num_examples: int,
        K_for_retrieval_eval: int,
    ):
        model = model.eval().to(DEVICE)
        final_results = dict()
        self.__evaluate_similarity(
            final_results, model, pairwise_similarity_eval_num_examples
        )
        self.__evaluate_retrieval(
            final_results,
            model,
            retrieval_eval_num_examples,
            K_for_retrieval_eval,
        )
        return final_results

    def __evaluate_similarity(
        self,
        final_results: dict[str, Any],
        model: GBDSim,
        pairwise_similarity_eval_num_examples: int,
    ):
        pairwise_eval_examples = [
            generate_synthetic_causal_data_example()
            for _ in range(pairwise_similarity_eval_num_examples)
        ]
        predictions = []
        labels = []
        for X1, y1, X2, y2, label in tqdm(pairwise_eval_examples):
            predictions.append(
                float(
                    model(
                        X1.to(DEVICE),
                        y1.to(DEVICE),
                        X2.to(DEVICE),
                        y2.to(DEVICE),
                    )
                )
            )
            labels.append(label)
        final_results["mae_by_model"] = mean_absolute_error(
            labels, predictions
        )
        final_results["mae_by_median"] = mean_absolute_error(
            labels, np.ones_like(labels) * np.median(labels)
        )
        final_results["spearmanr_corr"] = spearmanr(labels, predictions).statistic  # type: ignore # noqa: E501

    def __evaluate_retrieval(
        self,
        final_results: dict[str, Any],
        model: GBDSim,
        retrieval_eval_num_examples: int,
        K_for_retrieval_eval: int,
    ):
        retrieval_eval_graphs = [
            MlpCausalGraphFractory.generate_causal_graph()
            for _ in range(retrieval_eval_num_examples)
        ]
        retrieval_eval_datasets = [
            g.generate_data() for g in retrieval_eval_graphs
        ]
        real_distances_matrix = np.array(
            [
                [
                    (
                        calculate_graph_distance(
                            retrieval_eval_graphs[i].nx_graph,
                            retrieval_eval_graphs[j].nx_graph,
                        )
                        if i <= j
                        else 0.0
                    )
                    for j in range(retrieval_eval_num_examples)
                ]
                for i in tqdm(range(retrieval_eval_num_examples))
            ]
        )
        real_distances_matrix += real_distances_matrix.T
        real_distances_matrix[np.diag_indices_from(real_distances_matrix)] = (
            np.inf
        )
        real_closest_neighbors = np.argsort(real_distances_matrix, 1)[
            :, :K_for_retrieval_eval
        ]

        estimated_distances_matrix = np.array(
            [
                [
                    (
                        float(
                            model(
                                retrieval_eval_datasets[i][0].to(DEVICE),
                                retrieval_eval_datasets[i][1].to(DEVICE),
                                retrieval_eval_datasets[j][0].to(DEVICE),
                                retrieval_eval_datasets[j][1].to(DEVICE),
                            )
                        )
                        if i <= j
                        else 0.0
                    )
                    for j in range(retrieval_eval_num_examples)
                ]
                for i in tqdm(range(retrieval_eval_num_examples))
            ]
        )
        estimated_distances_matrix += estimated_distances_matrix.T
        estimated_distances_matrix[
            np.diag_indices_from(estimated_distances_matrix)
        ] = np.inf
        real_closest_neighbors = np.argsort(real_distances_matrix, 1)[
            :, :K_for_retrieval_eval
        ]
        top_1_real_closest = real_closest_neighbors[:, 0]
        estimated_closest_neighbors = np.argsort(
            estimated_distances_matrix, 1
        )[:, :K_for_retrieval_eval]
        final_results["top_k_recall"] = (
            (estimated_closest_neighbors.T == top_1_real_closest)
            .T.any(axis=1)
            .mean()
            .item()
        )
        final_results["average_intersection_size"] = np.mean(
            [
                len(
                    set(real_closest_neighbors[i]).intersection(
                        estimated_closest_neighbors[i]
                    )
                )
                for i in range(retrieval_eval_num_examples)
            ]
        )

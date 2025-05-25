import json
import random
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import spearmanr
from torch import Tensor, cdist, tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from gbdsim.data.data_preprocessor import DataPreprocessor
from gbdsim.utils.protocols import DatasetDistanceCalculator


class TabrepoResults:

    def evaluate_model(
        self, model: DatasetDistanceCalculator, test_dataloader: DataLoader
    ) -> dict[str, Any]:
        logger.info("Calculating metric estimation results")
        metric_estimation_results = calculate_metric_estimation_results(
            model, test_dataloader
        )
        logger.info("Calculating pipeline selection results")
        pipeline_selection_results = calculate_pipeline_selection_results(
            model
        )
        return {
            "metric_estimation_results": metric_estimation_results,
            "pipeline_selection_results": pipeline_selection_results,
        }


def calculate_metric_estimation_results(
    model: DatasetDistanceCalculator, test_dataloader: DataLoader
) -> dict[str, Any]:
    predictions = []
    labels = []
    observations = list(chain(*test_dataloader))
    for X1, y1, X2, y2, label in tqdm(observations):
        with torch.no_grad():
            predictions.append(
                model.calculate_dataset_distance(
                    X1.to(model.device),  # type: ignore
                    y1.cuda(model.device),  # type: ignore
                    X2.cuda(model.device),  # type: ignore
                    y2.cuda(model.device),  # type: ignore
                )
                .detach()
                .cpu()
                .item()
            )
            labels.append(label)
    predictions = torch.tensor(predictions)
    labels = torch.stack(labels)
    return {
        "model_mae": (labels.flatten() - predictions.flatten())
        .abs()
        .mean()
        .item(),
        "spearmanr_corr_model": spearmanr(labels, predictions).statistic,  # type: ignore # noqa: E501
        "median_mae": (labels.flatten() - torch.median(labels))
        .abs()
        .mean()
        .item(),
    }


def calculate_pipeline_selection_results(model):
    logger.info("Loading pipeline selection evaluation data")
    with open("data/tabrepo/selected_pipelines.json", "r") as f:
        selected_pipelines = json.load(f)
    with open("data/tabrepo/split.json", "r") as f:
        splits = json.load(f)
    raw_ranks = pd.read_csv("data/tabrepo/raw_ranks.csv")
    train_ranks = raw_ranks.loc[raw_ranks.dataset_id.isin(splits["train"])]
    test_ranks = raw_ranks.loc[raw_ranks.dataset_id.isin(splits["test"])]
    datasets = [
        (
            int(path.stem),
            DataPreprocessor().preprocess_pandas_data(pd.read_csv(path)),
        )
        for path in Path("data/tabrepo/datasets").iterdir()
    ]
    train_datasets = list(filter(lambda d: d[0] in splits["train"], datasets))
    test_datasets = list(filter(lambda d: d[0] in splits["test"], datasets))

    logger.info("Calculating landmarkers baseline")
    landmarkers_results = evaluate_landmarkers_baseline(
        train_ranks, test_ranks, selected_pipelines
    )
    logger.info("Calculating random pipeline baseline")
    random_pipeline_results = evaluate_random_pipeline_baseline(
        train_ranks, test_ranks, selected_pipelines
    )
    logger.info("Calculating random dataset baseline")
    random_dataset_results = evaluate_random_dataset_baseline(
        train_ranks, test_ranks, selected_pipelines
    )
    logger.info("Calculating model-based selection")
    model_based_results = evaluate_model_based_selection(
        model, train_ranks, test_ranks, train_datasets, test_datasets
    )
    return {
        "landmarkers": landmarkers_results,
        "random_pipeline": random_pipeline_results,
        "random_dataset": random_dataset_results,
        "model_based": model_based_results,
    }


def calculate_distances_to_rows_based_on_euclidean_distance(
    X: pd.DataFrame,
    y: pd.DataFrame,
    columns_to_select: list[str] | None = None,
) -> Tensor:
    if columns_to_select is not None:
        X = X[columns_to_select]
        y = y[columns_to_select]
    return cdist(
        tensor(y.values.reshape(1, -1)),
        tensor(X.values),
    ).flatten()


def get_closest_dataset_idx(
    X: pd.DataFrame,
    y: pd.DataFrame,
    distance_calculator: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
) -> int:
    distances = distance_calculator(X, y)
    closest_distance_idx = np.argmin(distances)
    return closest_distance_idx.item()


def search_closest_by_index(
    target_dataset: pd.DataFrame,
    dataset_base: pd.DataFrame,
    distance_calculator: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
) -> float:
    closest_distance_idx = get_closest_dataset_idx(
        dataset_base, target_dataset, distance_calculator
    )
    closest_row = dataset_base.iloc[[closest_distance_idx]]
    closest_row_ranks = closest_row.iloc[:, 1:-1].values.flatten()
    best_pipeline_idx = np.argmin(closest_row_ranks)
    return best_pipeline_idx.item()


def search_random_pipeline(
    target_dataset: pd.DataFrame, dataset_base: pd.DataFrame, *args, **kwargs
) -> float:
    num_pipelines = dataset_base.iloc[:, 1:-1].shape[1]
    return random.randint(0, num_pipelines - 1)


def search_best_pipeline_from_random_dataset(
    target_dataset: pd.DataFrame,
    dataset_base: pd.DataFrame,
    *args,
    **kwargs,
) -> float:
    return search_closest_by_index(
        target_dataset,
        dataset_base,
        lambda X, y: np.random.uniform(0, 1, (X.shape[0],)),
    )


def evaluate_landmarkers_baseline(
    train_ranks: pd.DataFrame,
    test_ranks: pd.DataFrame,
    selected_pipelines: list[str],
) -> dict[str, Any]:
    ranks = []
    for idx, row in tqdm(list(test_ranks.iterrows())):
        row = test_ranks.loc[[idx]]
        best_pipeline_idx = search_closest_by_index(
            row[selected_pipelines],
            train_ranks,
            partial(  # type: ignore
                calculate_distances_to_rows_based_on_euclidean_distance,
                columns_to_select=selected_pipelines,
            ),
        )
        best_pipeline_rank = row.iloc[:, 1:-1].values.reshape(-1)[
            best_pipeline_idx
        ]  # type: ignore
        ranks.append(best_pipeline_rank)
    return {
        "mean": np.mean(ranks).item(),
        "std": np.std(ranks).item(),
    }


def evaluate_random_pipeline_baseline(
    train_ranks: pd.DataFrame,
    test_ranks: pd.DataFrame,
    selected_pipelines: list[str],
) -> dict[str, Any]:
    ranks = []
    for idx, row in tqdm(list(test_ranks.iterrows())):
        row = test_ranks.loc[[idx]]
        for _ in range(1000):
            best_pipeline_idx = search_random_pipeline(
                row[selected_pipelines],
                train_ranks,
            )
            best_pipeline_rank = row.iloc[:, 1:-1].values.reshape(-1)[
                best_pipeline_idx
            ]  # type: ignore
            ranks.append(best_pipeline_rank)
    return {
        "mean": np.mean(ranks).item(),
        "std": np.std(ranks).item(),
    }


def evaluate_random_dataset_baseline(
    train_ranks: pd.DataFrame,
    test_ranks: pd.DataFrame,
    selected_pipelines: list[str],
):
    ranks = []
    for idx, row in tqdm(list(test_ranks.iterrows())):
        row = test_ranks.loc[[idx]]
        for _ in range(1000):
            best_pipeline_idx = search_best_pipeline_from_random_dataset(
                row[selected_pipelines],
                train_ranks,
            )
            best_pipeline_rank = row.iloc[:, 1:-1].values.reshape(-1)[
                best_pipeline_idx
            ]  # type: ignore
            ranks.append(best_pipeline_rank)
    return {
        "mean": np.mean(ranks).item(),
        "std": np.std(ranks).item(),
    }


def evaluate_model_based_selection(
    model: DatasetDistanceCalculator,
    train_ranks: pd.DataFrame,
    test_ranks: pd.DataFrame,
    train_datasets: list[tuple[int, tuple[Tensor, Tensor]]],
    test_datasets: list[tuple[int, tuple[Tensor, Tensor]]],
) -> dict[str, Any]:

    ranks = []
    for dataset_id, (X, y) in tqdm(test_datasets):
        distances = []
        dids = []
        for train_dataset_id, (X_train, y_train) in train_datasets:
            dids.append(train_dataset_id)
            with torch.no_grad():
                if X_train.shape[1] == 0 or X.shape[1] == 0:
                    distances.append(float("inf"))
                else:
                    distances.append(
                        model.calculate_dataset_distance(
                            X_train.cuda(), y_train.cuda(), X.cuda(), y.cuda()
                        )
                        .detach()
                        .cpu()
                        .item()
                    )
        closest_dataset_did = dids[np.argmin(distances)]
        best_pipeline_idx = np.argmin(
            train_ranks.loc[
                train_ranks.dataset_id == closest_dataset_did
            ].values.flatten()[1:-1]
        )
        ranks.append(
            test_ranks.loc[
                test_ranks.dataset_id == dataset_id
            ].values.flatten()[1:-1][best_pipeline_idx]
        )
    return {
        "mean": np.mean(ranks).item(),
        "std": np.std(ranks).item(),
    }

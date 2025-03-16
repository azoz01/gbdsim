from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dataset2vec.utils import DataUtils
from numpy.typing import NDArray
from torch import Tensor

from gbdsim.data.data_preprocessor import DataPreprocessor


class DatasetsPairsGenerator:
    def __init__(
        self,
        datasets: list[Tensor],
    ):
        self.data = datasets
        self.n_datasets = len(datasets)
        self.Xs = [dataset[:, :-1] for dataset in datasets]
        self.ys = [dataset[:, -1].reshape(-1, 1) for dataset in datasets]

    @staticmethod
    def from_paths(paths: list[Path]) -> DatasetsPairsGenerator:
        preprocessor = DataPreprocessor()
        datasets = []
        for path in paths:
            df = pd.read_csv(path)
            X, y = preprocessor.preprocess_pandas_data(df)
            datasets.append(torch.concat([X, y.unsqueeze(1)], dim=1))
        return DatasetsPairsGenerator(datasets)

    def generate_pair_of_datasets_with_label(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        dataset_1_idx, dataset_2_idx = self.__get_random_datasets_indices()
        return (
            *self.__generate_dataset_subsample(dataset_1_idx),
            *self.__generate_dataset_subsample(dataset_2_idx),
            int(dataset_1_idx == dataset_2_idx),
        )

    def __get_random_datasets_indices(
        self,
    ) -> tuple[int, int]:
        if np.random.uniform() >= 0.5:
            idx = np.random.choice(self.n_datasets, 1)[0]
            return (idx, idx)
        else:
            idx1, idx2 = np.random.choice(
                self.n_datasets, 2, replace=False
            ).astype(int)
            return (idx1, idx2)

    def __generate_dataset_subsample(
        self, dataset_idx: int
    ) -> tuple[Tensor, Tensor]:
        X, y = self.Xs[dataset_idx], self.ys[dataset_idx]
        rows_idx, features_idx, targets_idx = self.__sample_batch_idx(X, y)
        X = DataUtils.index_tensor_using_lists(X, rows_idx, features_idx)
        y = DataUtils.index_tensor_using_lists(
            y, rows_idx, targets_idx
        ).reshape(-1)
        return X, y

    def __sample_batch_idx(
        self, X: Tensor, y: Tensor
    ) -> tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.generic]]:
        n_rows = X.shape[0]
        assert n_rows >= 8

        n_features = X.shape[1]
        n_targets = y.shape[1]

        max_q = min(int(np.log2(n_rows)), 8)
        q = np.random.choice(np.arange(3, max_q + 1))
        n_rows_to_select = 2**q
        rows_idx = np.random.choice(n_rows, n_rows_to_select)
        features_idx = DataUtils.sample_random_subset(n_features)
        targets_idx = DataUtils.sample_random_subset(n_targets)

        return rows_idx, features_idx, targets_idx

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from dataset2vec.utils import DataUtils
from numpy.typing import NDArray
from torch import Tensor

from gbdsim.data.data_preprocessor import DataPreprocessor


class DatasetsPairsWithLandmarkersGenerator:
    def __init__(self, datasets: list[Tensor], landmarkers: list[Tensor]):
        self.datasets = datasets
        self.landmarkers = landmarkers
        self.n_datasets = len(datasets)

    @staticmethod
    def from_paths(
        datasets_path: list[Path],
        performances_matrix_path: Path,
        chosen_landmarkers_path: Path,
        split: Literal["train", "test"],
    ) -> DatasetsPairsWithLandmarkersGenerator:
        with open("data/tabrepo/split.json", "r") as f:
            split_ids = json.load(f)[split]
        with open(chosen_landmarkers_path, "r") as f:
            chosen_landmarkers = json.load(f)

        preprocessor = DataPreprocessor()
        datasets = dict()
        for path in datasets_path:
            if int(path.stem) not in split_ids:
                continue
            df = pd.read_csv(path)
            X, y = preprocessor.preprocess_pandas_data(df)
            datasets[path.stem] = (X, y.unsqueeze(1))
        landmarkers = pd.read_csv(performances_matrix_path)
        landmarkers = landmarkers.loc[
            landmarkers.dataset_id.isin(split_ids),
            chosen_landmarkers + ["dataset_id"],
        ]
        landmarkers = {
            str(int(row[1].dataset_id.item())): Tensor(
                row[1][1:-1:].astype(float).values
            )
            for row in landmarkers.iterrows()
        }
        return DatasetsPairsWithLandmarkersGenerator(
            datasets=[datasets[str(id)] for id in split_ids],
            landmarkers=[landmarkers[str(id)] for id in split_ids],
        )

    def generate_pair_of_datasets_with_label(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        dataset_1_idx, dataset_2_idx = np.random.choice(
            self.n_datasets, 2, replace=False
        ).astype(int)
        if (
            self.datasets[dataset_1_idx][0].shape[0] == 0
            or self.datasets[dataset_2_idx][0].shape[0] == 0
            or self.datasets[dataset_1_idx][0].shape[1] == 0
            or self.datasets[dataset_2_idx][0].shape[1] == 0
        ):
            return self.generate_pair_of_datasets_with_label()
        landmarkers_1, landmarkers_2 = (
            self.landmarkers[dataset_1_idx],
            self.landmarkers[dataset_2_idx],
        )
        return (
            *self.__generate_dataset_subsample(dataset_1_idx),
            *self.__generate_dataset_subsample(dataset_2_idx),
            torch.sqrt(((landmarkers_1 - landmarkers_2) ** 2).sum()),
        )

    def __generate_dataset_subsample(
        self, dataset_idx: int
    ) -> tuple[Tensor, Tensor]:
        X, y = self.datasets[dataset_idx]
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

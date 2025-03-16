from __future__ import annotations

from typing import Protocol

from torch import Tensor
from typing_extensions import Self


class TorchModel(Protocol):
    def eval(self) -> Self: ...

    def to(self, device: str) -> Self: ...


class DatasetDistanceCalculator(TorchModel):

    def calculate_dataset_distance(
        self, X1: Tensor, y1: Tensor, X2: Tensor, y2: Tensor
    ) -> Tensor: ...


class DatasetRepresentationCalculator(TorchModel):

    def calculate_dataset_representation(
        self, X: Tensor, y: Tensor
    ) -> Tensor: ...


class OriginClassifier(DatasetRepresentationCalculator):

    def calculate_dataset_origin_probability(
        self, X1: Tensor, y1: Tensor, X2: Tensor, y2: Tensor
    ) -> Tensor: ...

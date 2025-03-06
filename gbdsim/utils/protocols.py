from typing import Protocol

from torch import Tensor


class DatasetDistanceCalculator(Protocol):

    def calculate_dataset_distance(
        self, X1: Tensor, y1: Tensor, X2: Tensor, y2: Tensor
    ) -> Tensor: ...

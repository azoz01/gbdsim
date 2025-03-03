from typing import Any, Mapping, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, MetricCollection

from ..model.gbdsim import GBDSim
from .metrics import BinaryCrossEntropy


class GBDSimSimilarityClassification(pl.LightningModule, GBDSim):

    def __init__(self, *args, **kwargs):
        pl.LightningModule.__init__(self)
        GBDSim.__init__(self, *args, **kwargs)
        self.train_metrics = self.__get_metrics("train")
        self.validation_metrics = self.__get_metrics("validation")
        self.test_metrics = self.__get_metrics("test")

    def __get_metrics(self, prefix: str) -> MetricCollection:
        return MetricCollection(
            {
                "cross_entropy": BinaryCrossEntropy(),
                "accuracy": Accuracy(task="binary"),
            },
            prefix=prefix,
        )

    def training_step(
        self,
        batch: list[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
        ],
    ) -> torch.Tensor | Mapping[str, Any] | None:
        return self.__process_batch(self.train_metrics, batch)

    def validation_step(
        self,
        batch: list[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
        ],
    ) -> torch.Tensor | Mapping[str, Any] | None:
        return self.__process_batch(self.validation_metrics, batch)

    def test_step(
        self,
        batch: list[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
        ],
    ) -> torch.Tensor | Mapping[str, Any] | None:
        return self.__process_batch(self.test_metrics, batch)

    def __process_batch(
        self,
        metrics: MetricCollection,
        batch: list[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
        ],
    ) -> torch.Tensor:
        predictions = []
        labels = []
        for X1, y1, X2, y2, label in batch:
            predictions.append(
                torch.exp(-GBDSim.forward(self, X1, y1, X2, y2))
            )
            labels.append(label)
        predictions = torch.stack(predictions)
        labels = torch.stack(labels)
        metrics(predictions, labels)
        pl.LightningModule.log_dict(self, metrics)
        return F.binary_cross_entropy(predictions, labels)

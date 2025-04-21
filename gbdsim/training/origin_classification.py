import warnings
from typing import Any, Mapping, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics import Accuracy, MetricCollection

from gbdsim.experiment_config import TrainingConfig
from gbdsim.utils.protocols import OriginClassifier

from ..utils.constants import DEVICE
from .metrics import BinaryCrossEntropy


class OriginClassificationLearner(pl.LightningModule):

    def __init__(
        self,
        model: OriginClassifier,
        training_config: TrainingConfig,
    ):
        pl.LightningModule.__init__(self)
        self.model = model
        self.training_config = training_config
        self.example_input_array = (
            torch.tensor(
                [
                    [2.0, 3.0],
                    [4.0, 5.0],
                    [4.0, 5.0],
                ]
            ),
            torch.tensor([1.0, 0.0, 1.0]),
            torch.tensor([[3.0, 4.0, 10.0], [5.0, 6.0, 11.0]]),
            torch.tensor([0.0, 1.0]),
        )
        self.train_metrics = self.__get_metrics("train/")
        self.validation_metrics = self.__get_metrics("val/")
        self.test_metrics = self.__get_metrics("test/")

    def __get_metrics(self, prefix: str) -> MetricCollection:
        return MetricCollection(
            {
                "cross_entropy": BinaryCrossEntropy(),
                "accuracy": Accuracy(task="binary"),
            },
            prefix=prefix,
        )

    def forward(
        self,
        X1: torch.Tensor,
        y1: torch.Tensor,
        X2: torch.Tensor,
        y2: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.calculate_dataset_origin_probability(X1, y1, X2, y2)

    def training_step(
        self,
        batch: list[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
        ],
    ) -> torch.Tensor | Mapping[str, Any] | None:
        loss = self.__process_batch(self.train_metrics, batch)
        self._log_gradients()
        return loss

    def on_train_epoch_end(self) -> None:
        self._log_learning_rate()
        self._log_weights()

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
                self.model.calculate_dataset_origin_probability(
                    X1.to(DEVICE),
                    y1.to(DEVICE),
                    X2.to(DEVICE),
                    y2.to(DEVICE),
                )
            )
            labels.append(label)
        predictions = torch.stack(predictions).flatten()
        labels = torch.Tensor(labels).to(DEVICE)
        metrics(predictions, labels)
        pl.LightningModule.log_dict(
            self, metrics, prog_bar=True, on_epoch=True, on_step=True
        )
        return F.binary_cross_entropy(predictions, labels, reduction="mean")

    def _log_learning_rate(self) -> None:
        optimizer = self.optimizers()
        if optimizer:
            for i, param_group in enumerate(optimizer.param_groups):  # type: ignore # noqa: E501
                group_name = param_group.get("name", f"group_{i}")
                lr = param_group["lr"]
                self.log(
                    f"lr/{group_name}",
                    lr,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

    def _log_gradients(self) -> None:
        for name, param in self.named_parameters():
            if param.grad is not None:
                with warnings.catch_warnings():
                    self.log(
                        f"gradients/{name}",
                        param.grad.norm(),
                        prog_bar=False,
                    )

    def _log_weights(self):
        logger = self.logger.experiment  # type: ignore
        for name, param in self.named_parameters():
            logger.add_histogram(f"weights/{name}", param, self.current_epoch)

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[dict[str, Any]]]:
        optimizer = optim.RAdam(
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, threshold_mode="abs", mode="max"
        )

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/accuracy",
                "frequency": 1,
            }
        ]

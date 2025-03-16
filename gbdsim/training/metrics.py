import torch
import torch.nn.functional as F
from torchmetrics import Metric


class BinaryCrossEntropy(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "ce_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "obs_counter",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        self.ce_sum += F.binary_cross_entropy(preds, target, reduction="sum")
        self.obs_counter += preds.shape[0]

    def compute(self) -> torch.Tensor:
        return self.ce_sum.float() / self.obs_counter

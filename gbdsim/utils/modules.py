import torch
import torch.nn as nn


class EuclideanDistance(nn.Module):
    def __init__(self):
        super(EuclideanDistance, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=-1))


class SimilarityInputProcessor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(SimilarityInputProcessor, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.concat([x1, x2, x1 * x2], dim=1)


class MultiInputSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *inputs):
        x = inputs
        for layer in self.layers:
            if isinstance(x, tuple):
                x = layer(*x)
            else:
                x = layer(x)
        return x

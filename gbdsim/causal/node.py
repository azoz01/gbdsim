from typing import Callable

import torch


class CausalNode:

    def __init__(
        self,
        activation_function: Callable[[torch.Tensor], torch.Tensor],
        noise_std: float,
    ):
        self.activation_function = activation_function
        self.noise_std = noise_std
        self.noise_distribution = torch.distributions.normal.Normal(
            0, noise_std
        )

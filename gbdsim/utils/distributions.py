import math

import torch


def generate_tnlu(
    min_mu: float,
    max_mu: float,
    min_val: float,
    max_val: float,
    round_output: bool = True,
) -> float | int:
    dist = torch.distributions.uniform.Uniform(
        math.log(min_mu), math.log(max_mu)
    )
    mu, sd = torch.exp(dist.sample([2]))
    truncated = max(
        min(
            torch.distributions.normal.Normal(mu, sd).sample().item(),
            max_val,
        ),
        min_val,
    )
    if round_output:
        truncated = round(truncated)
    return truncated

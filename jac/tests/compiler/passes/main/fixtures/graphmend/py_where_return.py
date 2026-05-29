"""Python-source fixture: data-dependent return branches -> torch.where."""

import torch


@torch.compile()
def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        return x * (b * -1)
    else:
        return x * (b * -2)

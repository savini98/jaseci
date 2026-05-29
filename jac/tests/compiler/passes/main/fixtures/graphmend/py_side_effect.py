"""Python-source fixture: print is buffered and flushed after compute."""

import torch


@torch.compile()
def f(x: torch.Tensor) -> torch.Tensor:
    x = torch.relu(x)
    print(x)
    return torch.sin(x) * 2.0

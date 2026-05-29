"""Call-site entry point: an nn.Module wrapped with torch.compile(model) (not a
decorator). The break is in an undecorated method; entry-point analysis must tag
the wrapped class's methods. Mirrors how real HF models apply torch.compile."""

import torch


class MyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(x)
        if x.sum() > 0:
            return x * 2.0
        else:
            return x * 3.0


model = MyModel()
compiled = torch.compile(model)

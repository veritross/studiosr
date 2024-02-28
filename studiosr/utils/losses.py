import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduce = torch.mean if reduction == "mean" else torch.sum

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.reduce(torch.sqrt(torch.square(x - y) + self.eps))

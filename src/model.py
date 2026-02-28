"""
Flexible neural network for distributed training experiments.
Supports tabular data (any number of input features) with configurable depth.
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class TabularNet(nn.Module):
    """
    Configurable feed-forward network for tabular classification/regression.
    Designed for scalable distributed training with PyTorch DDP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout: float = 0.3,
        task: str = "binary",  # "binary", "multiclass", "regression"
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.task = task

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.head(features)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        if self.task == "binary":
            return torch.sigmoid(logits)
        elif self.task == "multiclass":
            return torch.softmax(logits, dim=-1)
        return logits  # regression

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""
Data loader with efficient preprocessing and dataset creation.
Supports sklearn datasets + custom CSV datasets.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import (
    make_classification, make_regression,
    fetch_covtype, load_breast_cancer
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class TorchDataset(Dataset):
    """Generic tensor dataset for tabular data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataset(
    dataset_name: str = "synthetic",
    n_samples: int = 100_000,
    n_features: int = 50,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[TorchDataset, TorchDataset, TorchDataset, int, int]:
    """
    Load and preprocess a dataset.

    Returns: train_ds, val_ds, test_ds, input_dim, output_dim
    """
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "synthetic":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=20,
            n_redundant=10,
            n_classes=2,
            random_state=random_state,
        )
        output_dim = 1

    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target.astype(float)
        output_dim = 1

    elif dataset_name == "covertype":
        data = fetch_covtype()
        X, y = data.data, (data.target == 1).astype(float)  # binary
        output_dim = 1

    elif dataset_name == "synthetic_large":
        X, y = make_classification(
            n_samples=500_000,
            n_features=100,
            n_informative=40,
            n_redundant=20,
            random_state=random_state,
        )
        output_dim = 1

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: synthetic, breast_cancer, covertype, synthetic_large")

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y if output_dim == 1 else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio / (1 - test_ratio),
        random_state=random_state,
    )

    logger.info(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    logger.info(f"  Input dim: {X.shape[1]} | Output dim: {output_dim}")

    return (
        TorchDataset(X_train, y_train),
        TorchDataset(X_val, y_val),
        TorchDataset(X_test, y_test),
        X.shape[1],
        output_dim,
    )


def make_loaders(
    train_ds: TorchDataset,
    val_ds: TorchDataset,
    test_ds: TorchDataset,
    batch_size: int = 512,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders, optionally with DistributedSampler."""

    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

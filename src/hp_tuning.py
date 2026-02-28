"""
Hyperparameter tuning using Ray Tune with Bayesian optimization.
Automatically finds the best configuration for the distributed trainer.

GCP equivalent: Vertex AI Vizier (hyperparameter tuning service)
"""

import sys
import os
import logging
import json
from pathlib import Path
from functools import partial

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Try importing Ray Tune (optional but preferred) ─────────────────── #
try:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray Tune not available. Falling back to grid search.")


def _objective_no_ray(config: dict) -> float:
    """Simple objective using plain PyTorch (no Ray)."""
    from src.trainer import train_worker
    result = train_worker(
        rank=0, world_size=1, config=config,
        run_name=f"hp_trial_{hash(str(config)) % 10000}",
        use_ddp=False,
    )
    return result or 0.0


def _ray_trainable(config: dict):
    """Ray Tune trainable function (reports AUC each epoch)."""
    from src.trainer import train_worker
    auc = train_worker(
        rank=0, world_size=1, config=config,
        run_name=f"ray_trial",
        use_ddp=False,
    )
    tune.report(val_auc=auc or 0.0)


def run_ray_tune(num_trials: int = 10, dataset: str = "synthetic") -> dict:
    """
    Run hyperparameter search using Ray Tune + Optuna Bayesian search.
    """
    search_space = {
        "dataset": dataset,
        "epochs": 15,
        "batch_size": tune.choice([256, 512, 1024]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "dropout": tune.uniform(0.1, 0.5),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "hidden_dims": tune.choice([
            [128, 64],
            [256, 128, 64],
            [512, 256, 128],
            [512, 256, 128, 64],
        ]),
        "patience": 5,
    }

    scheduler = ASHAScheduler(
        metric="val_auc",
        mode="max",
        max_t=15,
        grace_period=5,
        reduction_factor=2,
    )

    search_alg = OptunaSearch(metric="val_auc", mode="max")

    analysis = tune.run(
        _ray_trainable,
        config=search_space,
        num_samples=num_trials,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={"cpu": 2},
        name="hp_search",
        local_dir=str(CHECKPOINT_DIR),
        verbose=1,
    )

    best_config = analysis.best_config
    best_auc = analysis.best_result["val_auc"]
    logger.info(f"Best AUC: {best_auc:.4f}")
    logger.info(f"Best config: {best_config}")

    results = {"best_config": best_config, "best_val_auc": best_auc}
    with open(CHECKPOINT_DIR / "best_hp_config.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_config


def run_grid_search(dataset: str = "synthetic") -> dict:
    """
    Fallback grid search when Ray Tune is not available.
    Searches over a small predefined grid.
    """
    import mlflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("distributed-ml-training")

    grid = [
        {"lr": 1e-3, "dropout": 0.2, "hidden_dims": [256, 128, 64], "batch_size": 512},
        {"lr": 3e-4, "dropout": 0.3, "hidden_dims": [512, 256, 128], "batch_size": 512},
        {"lr": 5e-4, "dropout": 0.3, "hidden_dims": [512, 256, 128, 64], "batch_size": 256},
        {"lr": 1e-3, "dropout": 0.4, "hidden_dims": [256, 128], "batch_size": 1024},
    ]

    best_auc = 0.0
    best_config = None

    for i, overrides in enumerate(grid):
        config = {
            "dataset": dataset,
            "epochs": 15,
            "patience": 5,
            "weight_decay": 1e-4,
            **overrides
        }
        logger.info(f"\n=== Grid Search Trial {i+1}/{len(grid)} ===")
        logger.info(f"Config: {config}")

        auc = _objective_no_ray(config)
        logger.info(f"Trial {i+1} AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_config = config.copy()

    logger.info(f"\nBest AUC: {best_auc:.4f}")
    logger.info(f"Best config: {best_config}")

    results = {"best_config": best_config, "best_val_auc": best_auc}
    with open(CHECKPOINT_DIR / "best_hp_config.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_config


def run_hp_tuning(num_trials: int = 10, dataset: str = "synthetic") -> dict:
    """Entry point: run tuning with best available backend."""
    if RAY_AVAILABLE:
        logger.info("Using Ray Tune + Optuna for hyperparameter search.")
        return run_ray_tune(num_trials=num_trials, dataset=dataset)
    else:
        logger.info("Using manual grid search (install ray[tune] for Bayesian search).")
        return run_grid_search(dataset=dataset)


if __name__ == "__main__":
    best = run_hp_tuning(num_trials=8, dataset="synthetic")
    print(f"\nBest config found:\n{json.dumps(best, indent=2)}")

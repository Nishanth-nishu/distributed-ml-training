"""
Main experiment launcher for distributed ML training.
Runs experiments: single-node, multi-process DDP, and HP search.
"""

import sys
import os
import json
import logging
import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.trainer import train_worker
from src.hp_tuning import run_hp_tuning

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_single(config: dict, run_name: str):
    """Single-process training (CPU or 1 GPU)."""
    logger.info(f"Starting single-process run: {run_name}")
    return train_worker(rank=0, world_size=1, config=config, run_name=run_name, use_ddp=False)


def run_ddp(config: dict, run_name: str, world_size: int):
    """Multi-process DDP training."""
    logger.info(f"Starting DDP run with world_size={world_size}: {run_name}")
    mp.spawn(
        train_worker,
        args=(world_size, config, run_name, True),
        nprocs=world_size,
        join=True,
    )


def main():
    parser = argparse.ArgumentParser("Distributed ML Training Experiment Runner")
    parser.add_argument("--mode", choices=["single", "ddp", "hp_search", "benchmark"], default="single")
    parser.add_argument("--run-name", type=str, default="experiment")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "breast_cancer", "covertype", "synthetic_large"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--world-size", type=int, default=2,
                        help="Number of processes for DDP (auto-capped to available CPUs/GPUs)")
    parser.add_argument("--hp-trials", type=int, default=10)
    parser.add_argument("--config-file", type=str, default=None,
                        help="Path to JSON config file (overrides CLI args)")
    args = parser.parse_args()

    # Load config: file takes precedence over CLI args
    if args.config_file:
        with open(args.config_file) as f:
            config = json.load(f)
        logger.info(f"Loaded config from {args.config_file}")
    else:
        config = {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dropout": 0.3,
            "weight_decay": 1e-4,
            "hidden_dims": [512, 256, 128],
            "patience": 5,
        }

    # Determine available parallelism
    n_gpus = torch.cuda.device_count()
    n_cpus = os.cpu_count() or 1
    effective_world_size = min(args.world_size, max(n_gpus, 1))

    logger.info(f"Mode: {args.mode}")
    logger.info(f"GPUs: {n_gpus} | CPUs: {n_cpus} | Effective world_size: {effective_world_size}")
    logger.info(f"Config: {config}")

    if args.mode == "single":
        run_single(config, run_name=args.run_name)

    elif args.mode == "ddp":
        if effective_world_size < 2:
            logger.warning("Only 1 device available — falling back to single-process.")
            run_single(config, run_name=args.run_name)
        else:
            run_ddp(config, run_name=args.run_name, world_size=effective_world_size)

    elif args.mode == "hp_search":
        best_config = run_hp_tuning(num_trials=args.hp_trials, dataset=args.dataset)
        logger.info("HP search complete. Training best config...")
        run_single(best_config, run_name=f"{args.run_name}_best")

    elif args.mode == "benchmark":
        # Run 3 experiments and compare
        experiments = [
            ("baseline", {"hidden_dims": [128, 64], "lr": 1e-3, "batch_size": 512}),
            ("medium",   {"hidden_dims": [256, 128, 64], "lr": 3e-4, "batch_size": 512}),
            ("large",    {"hidden_dims": [512, 256, 128, 64], "lr": 3e-4, "batch_size": 256}),
        ]
        results = {}
        for exp_name, overrides in experiments:
            full_config = {**config, **overrides}
            logger.info(f"\n{'='*50}")
            logger.info(f"Running benchmark: {exp_name}")
            auc = run_single(full_config, run_name=f"{args.run_name}_{exp_name}")
            results[exp_name] = auc

        logger.info("\n=== Benchmark Results ===")
        for name, auc in results.items():
            logger.info(f"  {name:15s}: AUC={auc:.4f}")


if __name__ == "__main__":
    main()

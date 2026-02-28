"""
Core distributed trainer using PyTorch DDP (Data Distributed Parallel).
Supports both single-GPU and multi-GPU training with MLflow tracking.

GCP equivalent: Vertex AI custom training jobs with distributed strategy
"""

import os
import sys
import time
import logging
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import TabularNet
from src.data_loader import get_dataset, make_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [rank %(process)d] %(message)s"
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def setup_ddp(rank: int, world_size: int, backend: str = "gloo"):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.manual_seed(42 + rank)
    logger.info(f"DDP initialized: rank={rank}, world_size={world_size}, backend={backend}")


def cleanup_ddp():
    dist.destroy_process_group()


def compute_metrics(all_preds: list, all_labels: list) -> Dict[str, float]:
    preds_np = np.array(all_preds)
    labels_np = np.array(all_labels)
    binary_preds = (preds_np > 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(labels_np, preds_np)),
        "accuracy": float(accuracy_score(labels_np, binary_preds)),
        "f1": float(f1_score(labels_np, binary_preds, zero_division=0)),
    }


@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X).squeeze(-1)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = avg_loss
    return metrics


def train_worker(
    rank: int,
    world_size: int,
    config: Dict[str, Any],
    run_name: str,
    use_ddp: bool = True,
):
    """
    Main training worker — runs on each GPU/process.
    rank=0 is the primary process that logs to MLflow.
    """
    is_master = (rank == 0)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if use_ddp and world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        setup_ddp(rank, world_size, backend)

    # ── Data ─────────────────────────────────────────────────────────── #
    train_ds, val_ds, test_ds, input_dim, output_dim = get_dataset(
        dataset_name=config.get("dataset", "synthetic"),
        n_samples=config.get("n_samples", 100_000),
    )

    distributed = use_ddp and world_size > 1
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds,
        batch_size=config.get("batch_size", 512),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    # ── Model ─────────────────────────────────────────────────────────── #
    model = TabularNet(
        input_dim=input_dim,
        hidden_dims=config.get("hidden_dims", [512, 256, 128]),
        output_dim=output_dim,
        dropout=config.get("dropout", 0.3),
    ).to(device)

    if use_ddp and world_size > 1:
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    if is_master:
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Optimizer & Loss ─────────────────────────────────────────────── #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 3e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.get("lr", 3e-4),
        steps_per_epoch=len(train_loader),
        epochs=config.get("epochs", 20),
    )
    criterion = nn.BCEWithLogitsLoss()

    # ── MLflow tracking (master process only) ────────────────────────── #
    if is_master:
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("distributed-ml-training")
        mlflow_run = mlflow.start_run(run_name=run_name)
        mlflow.log_params({**config, "input_dim": input_dim, "world_size": world_size})

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config.get("epochs", 20) + 1):
        # Sync DDP sampler for correct shuffling per epoch
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        # TRAIN
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * len(y)

        train_loss = epoch_loss / len(train_loader.dataset)
        elapsed = time.time() - t0

        # EVAL (master only — avoids redundant computation)
        if is_master:
            val_metrics = evaluate(model, val_loader, criterion, device)
            val_loss = val_metrics["loss"]
            val_auc = val_metrics["auc"]
            val_f1 = val_metrics["f1"]
            val_acc = val_metrics["accuracy"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)
            history["val_f1"].append(val_f1)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_accuracy": val_acc,
                "epoch_time_s": elapsed,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            logger.info(
                f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | "
                f"AUC={val_auc:.4f} | F1={val_f1:.4f} | {elapsed:.1f}s"
            )

            # Checkpoint best model
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                raw_model = model.module if isinstance(model, DDP) else model
                best_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
                torch.save(best_state, CHECKPOINT_DIR / f"{run_name}_best.pt")
                logger.info(f"  ✓ Best AUC={best_auc:.4f} — checkpoint saved.")
            else:
                patience_counter += 1
                if patience_counter >= config.get("patience", 5):
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

    # FINAL TEST EVALUATION
    if is_master:
        raw_model = model.module if isinstance(model, DDP) else model
        if best_state:
            raw_model.load_state_dict(best_state)

        test_metrics = evaluate(raw_model, test_loader, criterion, device)
        logger.info(f"\nTest Results: AUC={test_metrics['auc']:.4f} | "
                    f"F1={test_metrics['f1']:.4f} | Acc={test_metrics['accuracy']:.4f}")

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.pytorch.log_model(raw_model, "model")

        # Save history
        results = {
            "run_name": run_name,
            "config": config,
            "history": history,
            "best_val_auc": best_auc,
            "test_metrics": test_metrics,
            "world_size": world_size,
        }
        results_path = CHECKPOINT_DIR / f"{run_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        mlflow.end_run()
        logger.info(f"Run complete. Results saved to {results_path}")

    if use_ddp and world_size > 1:
        cleanup_ddp()

    return best_auc if is_master else None

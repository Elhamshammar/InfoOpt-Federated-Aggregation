"""Counterfactual probe utilities for observational validation."""

import torch
import numpy as np
import time
from typing import Dict, List, Any


def create_validation_split(
    labels,
    val_ratio: float = 0.15,
    num_classes: int = None,  # accepted for compatibility, not used
    seed: int = 42,
):
    """Create a stratified validation split from training labels.
    
    Returns: val_indices as a numpy array.
    """
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=labels,
        random_state=seed,
    )
    return val_idx


def create_validation_loader(
    train_set,
    val_indices,
    batch_size: int,
    num_workers: int = 2,
):
    """Create a deterministic validation DataLoader.
    
    Uses shuffle=False and drop_last=False.
    """
    from torch.utils.data import Subset, DataLoader
    
    val_subset = Subset(train_set, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return val_loader


def evaluate_probe_candidates(
    probe_model: torch.nn.Module,
    fedavg_state: Dict[str, torch.Tensor],
    infoopt_state: Dict[str, torch.Tensor],
    val_loader: torch.utils.data.DataLoader,
    device: str,
):
    """Evaluate FedAvg and InfoOpt candidates on the same validation set.
    
    Uses a single reusable model to avoid RNG advancement and memory overhead.
    Both candidates use identical validation batches in identical order.
    
    Returns:
        (fa_metrics_dict, io_metrics_dict) where each is the output of evaluate().
    """
    from utils.metrics import evaluate
    
    probe_model.eval()
    
    # Evaluate FedAvg candidate
    probe_model.load_state_dict(fedavg_state, strict=True)
    fa_metrics = evaluate(probe_model, val_loader, device)
    
    # Evaluate InfoOpt candidate
    probe_model.load_state_dict(infoopt_state, strict=True)
    io_metrics = evaluate(probe_model, val_loader, device)
    
    return fa_metrics, io_metrics


def log_probe_results(
    probe_logger,
    round_num: int,
    fa_metrics: Dict[str, Any],
    io_metrics: Dict[str, Any],
    weight_l1_diff: float,
    candidate_state_l2: float,
    runtime_sec: float,
):
    """Log probe results to the dedicated CSV logger."""
    if probe_logger is None:
        return
    
    delta_loss = fa_metrics["loss"] - io_metrics["loss"]
    delta_acc = io_metrics["acc"] - fa_metrics["acc"]
    
    probe_logger.log({
        "round": round_num,
        "delta_val_loss": delta_loss,
        "delta_val_acc": delta_acc,
        "fa_val_loss": fa_metrics["loss"],
        "fa_val_acc": fa_metrics["acc"],
        "io_val_loss": io_metrics["loss"],
        "io_val_acc": io_metrics["acc"],
        "weight_l1_diff": weight_l1_diff,
        "candidate_state_l2_diff": candidate_state_l2,
        "probe_runtime_sec": runtime_sec,
    })

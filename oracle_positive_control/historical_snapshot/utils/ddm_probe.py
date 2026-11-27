"""DDM Stage 0A: Oracle magnitude probe.

Read-only diagnostic that evaluates hypothetical FedAvg magnitudes
without altering the training trajectory.
"""

import torch
import numpy as np
from typing import Dict, List
from utils.metrics import evaluate


DDM_GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def build_candidate_state(
    global_state: Dict[str, torch.Tensor],
    fedavg_state: Dict[str, torch.Tensor],
    gamma: float,
) -> Dict[str, torch.Tensor]:
    """Construct theta_t + gamma * (fedavg_state - global_state)."""
    candidate = {}
    for name in global_state.keys():
        # Only interpolate floating-point trainable parameters
        if torch.is_floating_point(global_state[name]):
            delta = fedavg_state[name] - global_state[name]
            candidate[name] = global_state[name] + gamma * delta
        else:
            # For non-trainable buffers, use the FedAvg result (conservative)
            candidate[name] = fedavg_state[name]
    return candidate


def validate_gamma1_equivalence(
    global_state: Dict[str, torch.Tensor],
    fedavg_state: Dict[str, torch.Tensor],
    tol: float = 1e-5,
) -> bool:
    """Verify that gamma=1 reproduces the FedAvg next state."""
    rel_errors = []
    for name in global_state.keys():
        if torch.is_floating_point(global_state[name]):
            expected = fedavg_state[name]
            actual = global_state[name] + 1.0 * (fedavg_state[name] - global_state[name])
            diff_norm = (actual - expected).abs().mean().item()
            base_norm = expected.abs().mean().item() + 1e-12
            rel_errors.append(diff_norm / base_norm)
    
    max_rel_err = max(rel_errors) if rel_errors else 0.0
    ok = max_rel_err < tol
    if not ok:
        print(f"[DDM WARNING] gamma=1 equivalence failed: max_rel_err={max_rel_err:.2e}")
    return ok


def evaluate_ddm_oracle(
    model: torch.nn.Module,
    global_state: Dict[str, torch.Tensor],
    fedavg_state: Dict[str, torch.Tensor],
    val_loader: torch.utils.data.DataLoader,
    device: str,
    round_idx: int,
) -> List[dict]:
    """Evaluate all DDM gamma candidates for a single round.

    Returns:
        list of dicts, one per gamma, with keys:
            round, gamma, val_loss, val_acc, macro_f1, rel_err_gamma1
    """
    if not validate_gamma1_equivalence(global_state, fedavg_state, tol=1e-5):
        raise RuntimeError(
            "DDM gamma=1 validation failed. Candidate construction is inconsistent "
            "with the actual FedAvg next state. Aborting probe."
        )

    records = []
    probe_model = model  # reuse; will load_state_dict for each candidate

    for gamma in DDM_GAMMAS:
        candidate = build_candidate_state(global_state, fedavg_state, gamma)

        # Double-check gamma=1
        if gamma == 1.0:
            # Verify candidate exactly matches fedavg_state
            rel_err = 0.0
            for name in candidate.keys():
                if torch.is_floating_point(candidate[name]):
                    diff = (candidate[name] - fedavg_state[name]).abs().max().item()
                    base = fedavg_state[name].abs().max().item() + 1e-12
                    rel_err = max(rel_err, diff / base)
        else:
            rel_err = float("nan")

        probe_model.load_state_dict(candidate, strict=True)
        metrics = evaluate(probe_model, val_loader, device)

        records.append({
            "round": round_idx,
            "gamma": gamma,
            "val_loss": metrics.get("loss", float("nan")),
            "val_acc": metrics.get("accuracy", float("nan")),
            "macro_f1": metrics.get("macro_f1", float("nan")),
            "rel_err_gamma1": rel_err,
        })

    # Restore the true FedAvg state so training continues unchanged
    probe_model.load_state_dict(fedavg_state, strict=True)

    return records

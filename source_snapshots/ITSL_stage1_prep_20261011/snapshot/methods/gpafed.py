"""GPAFed: Gradient Projection-Guided Adaptive Aggregation (IEEE TVT 2025).

Dynamic aggregation weights computed each round from gradient projection
contributions.

lambda_k = <g_global_prev, g_k> / ||g_global_prev||^2
alpha_k = softmax(lambda_k)   [or sample-size weighted]
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch


StateDict = Mapping[str, torch.Tensor]


def _flatten_model_update(
    global_state: StateDict,
    local_state: StateDict,
) -> torch.Tensor:
    """Effective local gradient direction: g_k ∝ w_global - w_local."""
    flattened = []
    for name, global_tensor in global_state.items():
        if name not in local_state:
            raise KeyError(f"Missing parameter in local state: {name}")
        if not torch.is_floating_point(global_tensor):
            continue
        update = (
            global_tensor.detach().to(torch.float64)
            - local_state[name].detach().to(torch.float64)
        )
        flattened.append(update.reshape(-1).cpu())
    if not flattened:
        raise ValueError("No floating-point parameters found.")
    return torch.cat(flattened)


def _projection_contribution(
    local_gradient: torch.Tensor,
    previous_global_gradient: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Equation (6): lambda_k = <g_prev, g_k> / ||g_prev||^2."""
    if local_gradient.shape != previous_global_gradient.shape:
        raise ValueError("Gradient vectors must have the same shape.")
    denominator = torch.dot(previous_global_gradient, previous_global_gradient)
    if float(denominator) <= eps:
        return 0.0
    numerator = torch.dot(previous_global_gradient, local_gradient)
    return float((numerator / denominator).item())


def _compute_weights(
    contributions: np.ndarray,
    sample_counts: np.ndarray,
    equal_size_tolerance: float = 1e-12,
) -> np.ndarray:
    """Equation (7): alpha_k = D_k * exp(lambda_k) / sum_j D_j * exp(lambda_j)."""
    contributions = np.asarray(contributions, dtype=np.float64)
    sample_counts = np.asarray(sample_counts, dtype=np.float64)

    if contributions.ndim != 1:
        raise ValueError("contributions must be one-dimensional.")
    if contributions.shape != sample_counts.shape:
        raise ValueError("contributions and sample_counts must have equal shape.")
    if np.any(sample_counts <= 0):
        raise ValueError("sample_counts must be positive.")

    equal_sizes = float(sample_counts.max() - sample_counts.min()) <= equal_size_tolerance

    shifted = contributions - np.max(contributions)
    exponentials = np.exp(shifted)

    if equal_sizes:
        unnormalized = exponentials
    else:
        unnormalized = sample_counts * exponentials

    if not np.isfinite(unnormalized).all() or unnormalized.sum() <= 0:
        unnormalized = sample_counts.copy()

    return unnormalized / unnormalized.sum()


class GPAFedController:
    """Computes dynamic GPAFed aggregation weights each round."""

    def __init__(self) -> None:
        self.previous_global_gradient: torch.Tensor | None = None

    def compute_weights(
        self,
        global_state: StateDict,
        local_states: Sequence[StateDict],
        sample_counts: Sequence[int],
    ) -> tuple[np.ndarray, dict]:
        """Compute round weights from gradient projections.

        Returns:
            weights: Normalized aggregation weights.
            diagnostics: Dict with contributions, weights, gradient norm.
        """
        local_gradients = [
            _flatten_model_update(global_state, local_state)
            for local_state in local_states
        ]

        if self.previous_global_gradient is None:
            reference_gradient = torch.stack(local_gradients, dim=0).mean(dim=0)
        else:
            reference_gradient = self.previous_global_gradient

        contributions = np.asarray(
            [
                _projection_contribution(g, reference_gradient)
                for g in local_gradients
            ],
            dtype=np.float64,
        )

        weights = _compute_weights(
            contributions=contributions,
            sample_counts=np.asarray(sample_counts),
        )

        current_global_gradient = torch.zeros_like(local_gradients[0])
        for w, g in zip(weights, local_gradients):
            current_global_gradient += float(w) * g

        self.previous_global_gradient = current_global_gradient.detach().clone()

        diagnostics = {
            "contributions": contributions.tolist(),
            "weights": weights.tolist(),
            "global_gradient_norm": float(
                torch.linalg.vector_norm(current_global_gradient).item()
            ),
        }

        return weights, diagnostics

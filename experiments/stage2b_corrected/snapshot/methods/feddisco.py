"""FedDisco: Discrepancy-Aware Collaboration in Federated Learning (ICML 2023).

Fixed aggregation weights computed once from class-distribution discrepancy.

p_k = ReLU(n_k - a*d_k + b) / sum_j ReLU(n_j - a*d_j + b)

where d_k is the discrepancy between client k's class distribution and the
uniform target distribution.
"""
from __future__ import annotations

import numpy as np


_EPS = 1e-12


def _normalize_distribution(values: np.ndarray, eps: float = _EPS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= eps:
        raise ValueError("Distribution has zero total mass.")
    return values / total


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = _EPS) -> float:
    p = _normalize_distribution(p)
    q = _normalize_distribution(q)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _l2_discrepancy(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize_distribution(p)
    q = _normalize_distribution(q)
    return float(np.linalg.norm(p - q, ord=2))


class FedDiscoController:
    """Computes fixed FedDisco aggregation weights once before training."""

    def __init__(self, num_classes: int, a: float = 0.5, b: float = 0.1, discrepancy_metric: str = "kl"):
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.discrepancy_metric = discrepancy_metric.lower()
        self.target = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
        self._client_weights: dict[int, float] | None = None
        self._diagnostics: dict | None = None

    def precompute_weights(self, client_label_hists: dict[int, np.ndarray], client_num_samples: dict[int, int]) -> dict[int, float]:
        """Compute fixed weights from label histograms and sample counts.

        Args:
            client_label_hists: Mapping client_id -> label histogram (length num_classes).
            client_num_samples: Mapping client_id -> number of samples.

        Returns:
            Mapping client_id -> aggregation weight.
        """
        client_ids = sorted(client_label_hists.keys())
        if not client_ids:
            raise ValueError("At least one client is required.")

        sample_counts = np.asarray([client_num_samples[cid] for cid in client_ids], dtype=np.float64)
        if np.any(sample_counts <= 0):
            raise ValueError("Every client must contain at least one sample.")

        relative_sizes = sample_counts / sample_counts.sum()

        discrepancies = []
        for cid in client_ids:
            local_dist = _normalize_distribution(client_label_hists[cid])
            if self.discrepancy_metric == "kl":
                d = _kl_divergence(local_dist, self.target)
            elif self.discrepancy_metric == "l2":
                d = _l2_discrepancy(local_dist, self.target)
            else:
                raise ValueError("discrepancy_metric must be 'kl' or 'l2'")
            discrepancies.append(d)

        discrepancies_np = np.asarray(discrepancies, dtype=np.float64)
        raw_weights = np.maximum(relative_sizes - self.a * discrepancies_np + self.b, 0.0)

        if not np.isfinite(raw_weights).all():
            raise FloatingPointError(f"Non-finite FedDisco raw weights: {raw_weights}")

        if raw_weights.sum() <= 1e-12:
            weights = relative_sizes.copy()
            fallback = True
        else:
            weights = raw_weights / raw_weights.sum()
            fallback = False

        self._client_weights = {cid: float(w) for cid, w in zip(client_ids, weights)}
        self._diagnostics = {
            "relative_sizes": relative_sizes.tolist(),
            "discrepancies": discrepancies_np.tolist(),
            "raw_weights": raw_weights.tolist(),
            "weights": weights.tolist(),
            "fallback_to_fedavg": fallback,
        }
        return self._client_weights

    def get_weights(self, selected_client_ids: list[int]) -> tuple[list[float], dict]:
        """Return normalized weights for selected clients.

        If some clients are not in the precomputed set, fall back to uniform.
        """
        if self._client_weights is None:
            raise RuntimeError("Weights not precomputed. Call precompute_weights first.")

        weights = []
        for cid in selected_client_ids:
            if cid in self._client_weights:
                weights.append(self._client_weights[cid])
            else:
                weights.append(0.0)

        weights_np = np.asarray(weights, dtype=np.float64)
        if weights_np.sum() <= 1e-12:
            weights_np = np.ones_like(weights_np) / len(weights_np)
        else:
            weights_np = weights_np / weights_np.sum()

        return weights_np.tolist(), self._diagnostics.copy()

    def diagnostics(self) -> dict | None:
        return self._diagnostics.copy() if self._diagnostics is not None else None

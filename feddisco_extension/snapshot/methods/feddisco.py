"""FedDisco: Discrepancy-Aware Collaboration in Federated Learning (ICML 2023).

Official-source-aligned implementation.  Weights are computed per round over
only the participating clients, using KL divergence normalized across those
participants, exactly as in the reference repository.

Weighting equation (paper Eq. 3):

    p_k = ReLU(n_k - a*d_k + b) / sum_j ReLU(n_j - a*d_j + b)

where
  * n_k is client k's dataset size relative to the participating cohort;
  * d_k is the discrepancy between client k's class distribution and the
    uniform target distribution, with KL values normalized across participants;
  * a, b are fixed hyper-parameters (defaults 0.5, 0.1).
"""
from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Tuple

import numpy as np
from scipy import special


_EPS = 1e-12


def _normalize_distribution(values: np.ndarray, eps: float = _EPS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= eps:
        raise ValueError("Distribution has zero total mass.")
    return values / total


def _kl_divergence_raw(p: np.ndarray, q: np.ndarray, eps: float = _EPS) -> float:
    """Standard KL divergence KL(p||q) with clipping for numerical stability."""
    p = _normalize_distribution(p)
    q = _normalize_distribution(q)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _l2_discrepancy(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize_distribution(p)
    q = _normalize_distribution(q)
    return float(np.linalg.norm(p - q, ord=2))


def _shuffle_seed(dataset: str, seed: int, round_idx: int) -> int:
    """Deterministic seed for the per-round discrepancy permutation RNG."""
    key = f"feddisco-shuffle-v1|{dataset}|{seed}|{round_idx}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


class FedDiscoController:
    """Computes FedDisco aggregation weights per round over selected clients.

    Aligns with the official repository (MediaBrain-SJTU/FedDisco):
      * relative sizes are computed over the participating cohort;
      * KL divergence values are normalized across participants;
      * ReLU and normalization follow disco.py:disco_weight_adjusting.
    """

    def __init__(
        self,
        num_classes: int,
        a: float = 0.5,
        b: float = 0.1,
        discrepancy_metric: str = "kl",
        dataset_name: str = "",
        seed: int = 0,
    ):
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.discrepancy_metric = discrepancy_metric.lower()
        self.dataset_name = dataset_name
        self.seed = seed
        self.target = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
        self._client_label_hists: Dict[int, np.ndarray] | None = None
        self._client_num_samples: Dict[int, int] | None = None

    def precompute_weights(
        self,
        client_label_hists: Dict[int, np.ndarray],
        client_num_samples: Dict[int, int],
    ) -> None:
        """Store label histograms and sample counts; no global weights are fixed."""
        if not client_label_hists:
            raise ValueError("At least one client is required.")
        if any(client_num_samples[cid] <= 0 for cid in client_label_hists):
            raise ValueError("Every client must contain at least one sample.")
        self._client_label_hists = {int(cid): np.asarray(hist, dtype=np.float64) for cid, hist in client_label_hists.items()}
        self._client_num_samples = {int(cid): int(n) for cid, n in client_num_samples.items()}

    def _discrepancy(self, local_hist: np.ndarray) -> float:
        local_dist = _normalize_distribution(local_hist)
        if self.discrepancy_metric == "kl":
            return _kl_divergence_raw(local_dist, self.target)
        if self.discrepancy_metric == "l2":
            return _l2_discrepancy(local_dist, self.target)
        raise ValueError(f"discrepancy_metric must be 'kl' or 'l2', got {self.discrepancy_metric}")

    def get_weights(
        self,
        selected_client_ids: List[int],
        round_idx: int | None = None,
        shuffle: bool = False,
    ) -> Tuple[List[float], dict]:
        """Return normalized weights for selected clients.

        Args:
            selected_client_ids: clients participating this round, in order.
            round_idx: current communication round (required if shuffle=True).
            shuffle: if True, permute discrepancy values among selected clients.

        Returns:
            weights: list of floats aligned with selected_client_ids.
            diagnostics: dict with discrepancies, raw_weights, weights, fallback.
        """
        if self._client_label_hists is None or self._client_num_samples is None:
            raise RuntimeError("Call precompute_weights before get_weights.")

        selected = [int(cid) for cid in selected_client_ids]
        sample_counts = np.asarray([self._client_num_samples[cid] for cid in selected], dtype=np.float64)
        if np.any(sample_counts <= 0):
            raise ValueError("Every selected client must contain at least one sample.")

        relative_sizes = sample_counts / sample_counts.sum()

        # Compute raw discrepancies.
        discrepancies = np.asarray(
            [self._discrepancy(self._client_label_hists[cid]) for cid in selected],
            dtype=np.float64,
        )

        # Official source normalizes KL across participating clients.
        if self.discrepancy_metric == "kl":
            total_disc = float(discrepancies.sum())
            if total_disc <= _EPS:
                discrepancies = np.zeros_like(discrepancies)
            else:
                discrepancies = discrepancies / total_disc

        original_discrepancies = discrepancies.copy()
        shuffle_order: List[int] | None = None

        if shuffle:
            if round_idx is None:
                raise ValueError("round_idx is required when shuffle=True")
            shuffle_rng = random.Random(_shuffle_seed(self.dataset_name, self.seed, round_idx))
            shuffle_order = list(range(len(selected)))
            shuffle_rng.shuffle(shuffle_order)
            discrepancies = discrepancies[shuffle_order]

        raw_weights = np.maximum(relative_sizes - self.a * discrepancies + self.b, 0.0)

        if not np.isfinite(raw_weights).all():
            raise FloatingPointError(f"Non-finite FedDisco raw weights: {raw_weights}")

        fallback = False
        if raw_weights.sum() <= _EPS:
            # Deterministic fallback to participating-client size weights.
            weights = relative_sizes.copy()
            fallback = True
        else:
            weights = raw_weights / raw_weights.sum()

        diagnostics = {
            "relative_sizes": relative_sizes.tolist(),
            "discrepancies": original_discrepancies.tolist(),
            "discrepancies_used": discrepancies.tolist(),
            "shuffle_order": shuffle_order,
            "raw_weights": raw_weights.tolist(),
            "weights": weights.tolist(),
            "fallback_used": int(fallback),
            "fallback_reason": "all_raw_weights_nonpositive" if fallback else "none",
            "a": self.a,
            "b": self.b,
            "metric": self.discrepancy_metric,
        }

        return weights.tolist(), diagnostics

    def diagnostics(self) -> dict | None:
        return None

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from methods.infoopt import _entropy_of, _js_and_grad, InfoOptController, solve_entropic_weights
from utils.info_metrics import js_divergence


class RegimeReliabilityController:
    """Regime- and reliability-aware decision layer for InfoOpt-Constrained.

    The controller decides how strongly the final aggregation should deviate from
    FedAvg toward InfoOpt. It estimates two factors:

    1. Regime: the mean client-to-global label-distribution divergence
       (Jensen-Shannon). High heterogeneity regimes are more likely to benefit
       from non-uniform aggregation.

    2. Reliability: the quality of the information-derived client ranking,
       measured by score separation (MAD), rank consistency (Spearman), and
       temporal consistency (L2 difference of normalized scores).

    The decision gate is a sigmoid over a linear combination of the normalized
    regime score, the reliability score, and their interaction. The raw gate is
    EMA-smoothed and then clipped by a safety check that ensures the blended
    weights do not deviate too far from the FedAvg baseline.

    **Safety redesign:** Constraints are defined relative to FedAvg:
        - JS(P_w || P_FA) <= epsilon_dev   (label-mixture deviation)
        - ENC(w) >= ENC(w_FA) - delta_ENC   (diversity preservation)
    This guarantees alpha=0 (pure FedAvg) is always feasible.
    """

    def __init__(
        self,
        cfg: dict,
        global_label_hist: np.ndarray,
        variant: str = "RA-InfoOpt",
    ):
        self.cfg = cfg.get("regime_reliability_controller", cfg.get("ra_infoopt", {}))
        if not self.cfg:
            self.cfg = {}

        self.variant = variant
        self.global_label_hist = np.asarray(global_label_hist, dtype=float) + 1e-12
        self.global_label_hist /= self.global_label_hist.sum()

        # Online heterogeneity normalization: rolling window quantiles
        self.h_window_size = int(self.cfg.get("heterogeneity_window", 20))
        self.h_lower_quantile = float(self.cfg.get("heterogeneity_lower_quantile", 0.10))
        self.h_upper_quantile = float(self.cfg.get("heterogeneity_upper_quantile", 0.90))
        self.h_history: list[float] = []

        # EMA gate state
        self.alpha_ema = 0.0
        self.prev_scores: dict[int, float] = {}

        # Parameters
        self.warmup = int(self.cfg.get("warmup_rounds", 5))
        self.alpha_ema_rho = float(self.cfg.get("smoothing", {}).get("alpha_ema", 0.8))
        self.beta_sep = float(self.cfg.get("reliability", {}).get("beta_separation", 0.4))
        self.beta_rank = float(self.cfg.get("reliability", {}).get("beta_rank", 0.3))
        self.beta_temp = float(self.cfg.get("reliability", {}).get("beta_temporal", 0.3))
        self.sep_scale = float(self.cfg.get("reliability", {}).get("separation_scale", 0.25))
        self.a_h = float(self.cfg.get("gate", {}).get("heterogeneity_weight", 0.25))
        self.a_r = float(self.cfg.get("gate", {}).get("reliability_weight", 0.25))
        self.a_i = float(self.cfg.get("gate", {}).get("interaction_weight", 0.50))
        self.bias = float(self.cfg.get("gate", {}).get("bias", 0.50))
        self.gamma = float(self.cfg.get("gate", {}).get("temperature", 6.0))
        self.eps = float(self.cfg.get("eps", 1e-12))
        self.num_bs_steps = int(self.cfg.get("safety", {}).get("alpha_binary_search_steps", 20))
        self.tol = float(self.cfg.get("safety", {}).get("numerical_tolerance", 1.0e-8))
        self.enforce_js = bool(self.cfg.get("safety", {}).get("enforce_js_after_blending", True))
        self.enforce_enc = bool(self.cfg.get("safety", {}).get("enforce_enc_after_blending", True))
        self.js_deviation_limit = float(self.cfg.get("safety", {}).get("js_deviation_limit", 0.05))
        self.enc_deviation_limit = float(self.cfg.get("safety", {}).get("enc_deviation_limit", 1.0))

        # Variant-specific overrides for ablation gates
        if variant == "RA-InfoOpt-Heterogeneity":
            self.a_h, self.a_r, self.a_i = 1.0, 0.0, 0.0
        elif variant == "RA-InfoOpt-Reliability":
            self.a_h, self.a_r, self.a_i = 0.0, 1.0, 0.0

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Robust z-score normalization (same as InfoOptController._z)."""
        x = np.asarray(scores, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return (x - med) / (mad + 1e-8)

    def _compute_heterogeneity(
        self,
        client_hists: np.ndarray,
        fedavg_weights: np.ndarray,
    ) -> float:
        """Mean client-to-global JS divergence.

        Args:
            client_hists: (C, K) array of per-client label histograms.
            fedavg_weights: (C,) FedAvg weights (sample-size normalized).

        Returns:
            H_t = mean_i JS(p_i || p_global) where p_global = w_FA @ P.
        """
        client_hists = np.asarray(client_hists, dtype=float)
        fedavg_weights = np.asarray(fedavg_weights, dtype=float)
        p_global = fedavg_weights @ client_hists
        p_global = p_global / (p_global.sum() + self.eps)
        h = 0.0
        for i in range(client_hists.shape[0]):
            h += js_divergence(client_hists[i], p_global)
        return float(h / client_hists.shape[0])

    def _update_h_range(self, h: float) -> None:
        self.h_history.append(float(h))
        if len(self.h_history) > self.h_window_size:
            self.h_history.pop(0)

    def _normalize_heterogeneity(self, h: float) -> float:
        if len(self.h_history) < 5:
            # Not enough history: use theoretical range of pairwise JS.
            h_min_fallback = 0.0
            h_max_fallback = float(np.log(2.0))
            denom = h_max_fallback - h_min_fallback
            return float(np.clip((h - h_min_fallback) / denom, 0.0, 1.0))
        hist = np.asarray(self.h_history, dtype=float)
        h_min = float(np.quantile(hist, self.h_lower_quantile))
        h_max = float(np.quantile(hist, self.h_upper_quantile))
        denom = h_max - h_min
        if denom < 1e-12:
            return 0.5
        return float(np.clip((h - h_min) / denom, 0.0, 1.0))

    def _compute_separation(self, scores: np.ndarray) -> float:
        """MAD-based score separation reliability."""
        x = np.asarray(scores, dtype=float)
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med))) * 1.4826
        return float(mad / (mad + self.sep_scale))

    def _compute_rank_consistency(
        self,
        current_client_ids: list[int],
        current_scores: np.ndarray,
    ) -> float:
        """Spearman rank consistency between current and previous scores."""
        if not self.prev_scores:
            return 0.5

        common = [cid for cid in current_client_ids if cid in self.prev_scores]
        if len(common) < 3:
            return 0.5

        s_curr = np.array([current_scores[current_client_ids.index(cid)] for cid in common])
        s_prev = np.array([self.prev_scores[cid] for cid in common])

        if np.std(s_curr) < 1e-12 or np.std(s_prev) < 1e-12:
            return 0.5

        rho, _ = spearmanr(s_curr, s_prev)
        if np.isnan(rho):
            return 0.5
        return float((1.0 + rho) / 2.0)

    def _compute_temporal_consistency(
        self,
        current_client_ids: list[int],
        current_scores: np.ndarray,
    ) -> float:
        """L2 temporal consistency of normalized score vectors."""
        s_curr = self._normalize_scores(current_scores)
        s_prev = np.zeros_like(s_curr, dtype=float)
        for i, cid in enumerate(current_client_ids):
            if cid in self.prev_scores:
                s_prev[i] = (self.prev_scores[cid] - np.median(current_scores)) / (
                    np.median(np.abs(current_scores - np.median(current_scores))) + 1e-8
                )

        k = len(current_client_ids)
        diff = float(np.linalg.norm(s_curr - s_prev))
        return float(np.exp(-diff / (np.sqrt(k) + self.eps)))

    # ------------------------------------------------------------------
    # Phase 1: diagnostic helpers (read-only, do NOT modify state)
    # ------------------------------------------------------------------

    def _compute_robust_separation_diagnostic(self, scores: np.ndarray) -> tuple[float, float]:
        """MAD/median(|s|) separation and raw std. Returns (raw, std)."""
        x = np.asarray(scores, dtype=float)
        raw_std = float(np.std(x))
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * 1.4826
        median_abs = np.median(np.abs(x)) + self.eps
        raw = float(mad / median_abs)
        return raw, raw_std

    def _compute_rank_stability_diagnostic(
        self,
        current_client_ids: list[int],
        current_scores: np.ndarray,
    ) -> tuple[int, float, float, bool]:
        """Spearman rank stability over overlapping clients.
        Returns (overlap_count, rho_raw, rho_mapped_0_1, is_valid).
        Missing = NaN."""
        if not self.prev_scores:
            return 0, float("nan"), float("nan"), False

        curr_dict = dict(zip(current_client_ids, current_scores))
        overlap = [cid for cid in current_client_ids if cid in self.prev_scores]
        n_overlap = len(overlap)
        if n_overlap < 3:
            return n_overlap, float("nan"), float("nan"), False

        s_curr = np.array([curr_dict[cid] for cid in overlap])
        s_prev = np.array([self.prev_scores[cid] for cid in overlap])

        if np.std(s_curr) < 1e-12 or np.std(s_prev) < 1e-12:
            return n_overlap, float("nan"), float("nan"), False

        rho, _ = spearmanr(s_curr, s_prev)
        if np.isnan(rho):
            return n_overlap, float("nan"), float("nan"), False

        return n_overlap, float(rho), float((1.0 + rho) / 2.0), True

    def _compute_score_persistence_diagnostic(
        self,
        current_client_ids: list[int],
        current_scores: np.ndarray,
    ) -> tuple[float, bool]:
        """Pooled-normalized L2 persistence over overlapping clients.
        Read-only: does NOT modify self.prev_scores."""
        s_curr = np.asarray(current_scores, dtype=float)
        s_prev = np.zeros_like(s_curr, dtype=float)
        valid = np.zeros(len(current_client_ids), dtype=bool)

        for i, cid in enumerate(current_client_ids):
            if cid in self.prev_scores:
                s_prev[i] = self.prev_scores[cid]
                valid[i] = True

        n = int(valid.sum())
        if n < 2:
            return float("nan"), False

        pooled = np.concatenate([s_curr[valid], s_prev[valid]])
        mu = np.mean(pooled)
        sigma = np.std(pooled) + self.eps
        z_curr = (s_curr[valid] - mu) / sigma
        z_prev = (s_prev[valid] - mu) / sigma

        diff = float(np.linalg.norm(z_curr - z_prev))
        persist = float(np.exp(-diff / (np.sqrt(n) + self.eps)))
        return persist, True

    def _compute_blend_identity_diagnostic(
        self,
        w_blend: np.ndarray,
        w_fedavg: np.ndarray,
        w_infoopt: np.ndarray,
        alpha: float,
    ) -> tuple[float, float, float]:
        """Verify w_blend = (1-alpha) w_fa + alpha w_io.
        Returns (observed_blend_fraction, normalized_error, absolute_error)."""
        d_io_fa = float(np.sum(np.abs(w_infoopt - w_fedavg)))
        d_blend_fa = float(np.sum(np.abs(w_blend - w_fedavg)))
        if d_io_fa < 1e-12:
            return float("nan"), 0.0, 0.0
        frac = d_blend_fa / d_io_fa
        norm_err = abs(frac - alpha)
        abs_err = abs(d_blend_fa - alpha * d_io_fa)
        return frac, norm_err, abs_err
    # ------------------------------------------------------------------


    def _compute_reliability(
        self,
        current_client_ids: list[int],
        current_scores: np.ndarray,
    ) -> tuple[float, float, float, float]:
        r_sep = self._compute_separation(current_scores)
        r_rank = self._compute_rank_consistency(current_client_ids, current_scores)
        r_temp = self._compute_temporal_consistency(current_client_ids, current_scores)
        r = (
            self.beta_sep * r_sep
            + self.beta_rank * r_rank
            + self.beta_temp * r_temp
        )
        return float(r_sep), float(r_rank), float(r_temp), float(r)

    def _sigmoid(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _compute_gate(self, h_norm: float, r: float) -> float:
        logit = (
            self.a_h * h_norm
            + self.a_r * r
            + self.a_i * h_norm * r
            - self.bias
        )
        return self._sigmoid(self.gamma * logit)

    def _is_feasible(
        self,
        alpha: float,
        w_fedavg: np.ndarray,
        w_infoopt: np.ndarray,
        P: np.ndarray,
        p_fedavg_mixture: np.ndarray,
        enc_fedavg: float,
    ) -> tuple[bool, str]:
        """Check feasibility of a blend at alpha.

        Constraints (relative to FedAvg):
            JS(P_blend || P_FA) <= epsilon_dev
            ENC(w_blend) >= ENC(w_FA) - delta_ENC

        Returns:
            (feasible, reason)
        """
        w = (1.0 - alpha) * w_fedavg + alpha * w_infoopt
        w = np.clip(w, self.tol, None)
        w = w / w.sum()

        # JS deviation from FedAvg mixture
        if self.enforce_js:
            js_dev, _ = _js_and_grad(w, P, p_fedavg_mixture)
            if js_dev > self.js_deviation_limit + self.tol:
                return False, "js_violation"

        # ENC deviation from FedAvg
        if self.enforce_enc:
            enc_final = float(np.exp(_entropy_of(w)))
            if enc_final < enc_fedavg - self.enc_deviation_limit - self.tol:
                return False, "enc_violation"

        return True, "none"

    def _find_safe_alpha(
        self,
        alpha_target: float,
        w_fedavg: np.ndarray,
        w_infoopt: np.ndarray,
        P: np.ndarray,
        p_fedavg_mixture: np.ndarray,
        enc_fedavg: float,
    ) -> tuple[float, str]:
        """Binary search for the largest feasible alpha in [0, alpha_target].

        Returns:
            (alpha_safe, reason)
        """
        if alpha_target <= 0.0:
            return 0.0, "none"

        feasible, _ = self._is_feasible(
            alpha_target, w_fedavg, w_infoopt, P, p_fedavg_mixture, enc_fedavg
        )
        if feasible:
            return alpha_target, "none"

        lo, hi = 0.0, alpha_target
        alpha_safe = 0.0
        for _ in range(self.num_bs_steps):
            mid = 0.5 * (lo + hi)
            feasible, _ = self._is_feasible(
                mid, w_fedavg, w_infoopt, P, p_fedavg_mixture, enc_fedavg
            )
            if feasible:
                alpha_safe = mid
                lo = mid
            else:
                hi = mid

        if alpha_safe <= self.tol:
            return 0.0, "search_failure"
        return alpha_safe, "blend_infeasible"

    def test_safety_grid(
        self,
        w_fedavg: np.ndarray,
        w_infoopt: np.ndarray,
        P: np.ndarray,
        p_fedavg_mixture: np.ndarray,
        enc_fedavg: float,
        alpha_max: float = 1.0,
        n: int = 21,
    ) -> None:
        """Print a grid of feasibility over alpha for debugging."""
        print("=== Safety Grid Test ===")
        for alpha in np.linspace(0.0, alpha_max, n):
            w = (1.0 - alpha) * w_fedavg + alpha * w_infoopt
            w = np.clip(w, self.tol, None)
            w = w / w.sum()
            js_dev, _ = _js_and_grad(w, P, p_fedavg_mixture)
            enc = float(np.exp(_entropy_of(w)))
            feasible, reason = self._is_feasible(
                alpha, w_fedavg, w_infoopt, P, p_fedavg_mixture, enc_fedavg
            )
            print(
                f"alpha={alpha:.4f}  "
                f"JS_dev={js_dev:.6f}  "
                f"ENC={enc:.4f}  "
                f"feasible={feasible}  "
                f"reason={reason}"
            )
        print("========================")

    def compute(
        self,
        round_idx: int,
        results,
        fedavg_weights: list[float] | np.ndarray,
        infoopt_weights: list[float] | np.ndarray,
        infoopt_info: dict,
    ) -> tuple[list[float], dict]:
        """Compute regime/reliability-aware blended weights.

        Args:
            round_idx: current round (1-indexed).
            results: list of ClientResult for selected clients.
            fedavg_weights: FedAvg sample-size weights.
            infoopt_weights: InfoOpt-Constrained weights.
            infoopt_info: dict returned by InfoOptController.compute_weights.

        Returns:
            (final_weights, ra_info_dict)
        """
        w_fedavg = np.asarray(fedavg_weights, dtype=float)
        w_infoopt = np.asarray(infoopt_weights, dtype=float)
        w_fedavg = w_fedavg / (w_fedavg.sum() + self.eps)
        w_infoopt = w_infoopt / (w_infoopt.sum() + self.eps)

        client_ids = [r.client_id for r in results]
        C = len(results)
        P = np.stack([r.label_hist.astype(float) + self.eps for r in results])
        P = P / P.sum(axis=1, keepdims=True)

        # FedAvg mixture (equals global distribution when w_FA = sample-size weights)
        p_fedavg_mixture = w_fedavg @ P
        p_fedavg_mixture = p_fedavg_mixture / (p_fedavg_mixture.sum() + self.eps)
        enc_fedavg = float(np.exp(_entropy_of(w_fedavg)))

        # Compute current scores and heterogeneity
        current_scores = np.array(infoopt_info.get("scores", [0.0] * C), dtype=float)
        h_raw = self._compute_heterogeneity(P, w_fedavg)
        self._update_h_range(h_raw)
        h_norm = self._normalize_heterogeneity(h_raw)

        r_sep, r_rank, r_temp, r = self._compute_reliability(client_ids, current_scores)

        # ---- Phase 1: instrumentation (read-only, no behavior change) ----
        robust_sep_raw, infoopt_score_std = self._compute_robust_separation_diagnostic(
            current_scores
        )
        overlap_count, rank_rho, rank_mapped, rank_valid = (
            self._compute_rank_stability_diagnostic(client_ids, current_scores)
        )
        score_persist, score_persist_valid = (
            self._compute_score_persistence_diagnostic(client_ids, current_scores)
        )
        # ------------------------------------------------------------------


        # Warm-up: always use FedAvg
        alpha_raw = 0.0
        if round_idx > self.warmup:
            alpha_raw = self._compute_gate(h_norm, r)

        # EMA smoothing
        alpha_ema = self.alpha_ema_rho * self.alpha_ema + (1.0 - self.alpha_ema_rho) * alpha_raw
        self.alpha_ema = alpha_ema

        # ---- Safety check with relative-to-FedAvg constraints ----
        # Check endpoints
        fa_feasible, fa_reason = self._is_feasible(
            0.0, w_fedavg, w_infoopt, P, p_fedavg_mixture, enc_fedavg
        )
        io_feasible, io_reason = self._is_feasible(
            1.0, w_fedavg, w_infoopt, P, p_fedavg_mixture, enc_fedavg
        )
        blend_feasible, blend_reason = self._is_feasible(
            alpha_ema, w_fedavg, w_infoopt, P, p_fedavg_mixture, enc_fedavg
        )

        # Debug grid for infeasible blends in early rounds
        if not blend_feasible and round_idx <= 20 and round_idx > self.warmup:
            print(f"DEBUG round={round_idx}: blend infeasible, running grid test")
            self.test_safety_grid(
                w_fedavg, w_infoopt, P, p_fedavg_mixture, enc_fedavg,
                alpha_max=float(alpha_ema), n=21
            )

        # Decision sequence
        alpha_safe = 0.0
        fallback_reason = "none"
        if blend_feasible:
            alpha_safe = alpha_ema
            fallback_reason = "none"
        elif fa_feasible:
            alpha_safe, search_reason = self._find_safe_alpha(
                alpha_ema, w_fedavg, w_infoopt, P, p_fedavg_mixture, enc_fedavg
            )
            fallback_reason = search_reason if search_reason != "none" else "blend_infeasible"
        elif io_feasible:
            alpha_safe = 1.0
            fallback_reason = "infoopt_fallback"
        else:
            # Neither endpoint feasible — should not happen with relative constraints
            alpha_safe = 0.0
            fallback_reason = "both_infeasible"

        # Blend
        w_final = (1.0 - alpha_safe) * w_fedavg + alpha_safe * w_infoopt
        w_final = np.clip(w_final, self.tol, None)
        w_final = w_final / w_final.sum()

        # ---- Phase 1: blend identity check ----
        blend_fraction, blend_error, blend_error_abs = (
            self._compute_blend_identity_diagnostic(
                w_final, w_fedavg, w_infoopt, alpha_safe
            )
        )
        # Note: if future projection stages are added, recompute here
        # ------------------------------------------------------------------


        # Update score history
        for cid, s in zip(client_ids, current_scores.tolist()):
            self.prev_scores[cid] = float(s)

        # Compute final diagnostics
        js_final_dev, _ = _js_and_grad(w_final, P, p_fedavg_mixture)
        js_fedavg_dev, _ = _js_and_grad(w_fedavg, P, p_fedavg_mixture)
        js_infoopt_dev, _ = _js_and_grad(w_infoopt, P, p_fedavg_mixture)

        enc_final = float(np.exp(_entropy_of(w_final)))
        enc_infoopt = float(np.exp(_entropy_of(w_infoopt)))

        wdist_fa_io = float(np.sum(np.abs(w_infoopt - w_fedavg)))
        wdist_final_fa = float(np.sum(np.abs(w_final - w_fedavg)))

        info = {
            "heterogeneity_raw": h_raw,
            "heterogeneity_normalized": h_norm,
            "reliability_separation": r_sep,
            "reliability_rank": r_rank,
            "reliability_temporal": r_temp,
            "reliability_total": r,
            "alpha_raw": alpha_raw,
            "alpha_ema": alpha_ema,
            "alpha_safe": alpha_safe,
            "js_deviation_limit": self.js_deviation_limit,
            "enc_deviation_limit": self.enc_deviation_limit,
            "js_fedavg_deviation": float(js_fedavg_dev),
            "js_infoopt_deviation": float(js_infoopt_dev),
            "js_blend_deviation": float(js_final_dev),
            "enc_fedavg": enc_fedavg,
            "enc_infoopt": enc_infoopt,
            "enc_final": enc_final,
            "fedavg_feasible": int(fa_feasible),
            "infoopt_feasible": int(io_feasible),
            "blend_feasible": int(blend_feasible),
            "fallback_reason": fallback_reason,
            "weight_l1_infoopt_vs_fedavg": wdist_fa_io,
            "weight_l1_final_vs_fedavg": wdist_final_fa,
            "fallback_used": int(fallback_reason != "none"),
            # Phase 1 instrumentation
            "infoopt_score_std": infoopt_score_std,
            "robust_separation_raw": robust_sep_raw,
            "overlap_count": overlap_count,
            "rank_rho": rank_rho,
            "rank_mapped": rank_mapped,
            "rank_valid": int(rank_valid),
            "score_persistence": score_persist,
            "score_persistence_valid": int(score_persist_valid),
            "blend_fraction": blend_fraction,
            "blend_identity_error": blend_error,
            "blend_identity_error_abs": blend_error_abs,
        }

        return w_final.tolist(), info

    def reset(self):
        """Reset controller state. Useful for multi-seed runs."""
        self.alpha_ema = 0.0
        self.prev_scores.clear()
        self.h_history.clear()


class RAInfoOptController:
    """Convenience wrapper that combines InfoOpt + RegimeReliabilityController.

    This class keeps the existing InfoOptController intact and adds the RA layer
    on top, so that the original algorithm remains reproducible.
    """

    def __init__(self, cfg: dict, global_label_hist: np.ndarray, variant: str = "RA-InfoOpt"):
        self.infoopt = InfoOptController(cfg, global_label_hist, variant="InfoOpt-Full")
        self.ra = RegimeReliabilityController(cfg, global_label_hist, variant=variant)

    def compute_weights(self, results, round_idx: int = 1):
        infoopt_weights, infoopt_info = self.infoopt.compute_weights(results)
        ns = np.array([r.num_samples for r in results], dtype=float)
        fedavg_weights = ns / (ns.sum() + 1e-12)
        final_weights, ra_info = self.ra.compute(
            round_idx=round_idx,
            results=results,
            fedavg_weights=fedavg_weights,
            infoopt_weights=infoopt_weights,
            infoopt_info=infoopt_info,
        )
        info = {**infoopt_info, **ra_info}
        self.last_weights = final_weights
        return final_weights, info

    def get_last_weights(self):
        return getattr(self, "last_weights", None)

    def should_stop(self, round_idx: int, acc: float, info: dict) -> bool:
        return self.infoopt.should_stop(round_idx, acc, info)

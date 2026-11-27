from __future__ import annotations
import numpy as np
from utils.info_metrics import js_divergence


def _entropy_of(w: np.ndarray) -> float:
    w = np.clip(w, 1e-12, 1.0)
    return float(-np.sum(w * np.log(w)))


def _js_and_grad(w: np.ndarray, P: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    """JS(P_mix(w) || q) and its gradient w.r.t. w, where P_mix = w @ P.

    Rows of P are per-client label distributions (each sums to 1).
    Uses the identity dJS/dP_c = 0.5 * log(P_c / M_c), M = (P_mix + q)/2,
    hence dJS/dw_i = 0.5 * <P_i, log(P_mix / M)>.
    """
    p_mix = np.clip(w @ P, 1e-12, None)
    p_mix = p_mix / p_mix.sum()
    m = 0.5 * (p_mix + q)
    js = 0.5 * float(np.sum(p_mix * np.log(p_mix / m))) + 0.5 * float(np.sum(q * np.log(q / m)))
    grad = 0.5 * (P @ np.log(p_mix / m))
    return js, grad


def stage1_shuffle_seed(dataset: str, seed: int, round_idx: int) -> int:
    """STAGE1: dedicated deterministic seed for the score permutation.

    sha256 of UTF-8 'stage1-score-shuffle-v1|dataset|seed|round', first 8
    digest bytes as an unsigned big-endian integer. Never uses Python's
    process-dependent hash or the training RNG.
    """
    import hashlib
    key = f"stage1-score-shuffle-v1|{dataset}|{seed}|{round_idx}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def solve_entropic_weights(
    scores: np.ndarray,
    tau: float,
    P: np.ndarray | None = None,
    q: np.ndarray | None = None,
    epsilon: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve  max_w  <w, s> + tau*H(w)   s.t.  w in simplex,
                                               JS(w @ P || q) <= epsilon (optional).

    Unconstrained case has the closed form  w = softmax(s / tau) -- returned
    exactly, no numerical solve.

    Constrained case: the objective is strictly concave and the constraint
    g(w) = JS(w @ P || q) is convex (JS of a linear mixture), so this is a
    small convex program; we solve it with SLSQP using analytic gradients.
    If the solver fails, a guaranteed-feasible fallback blends the
    unconstrained optimum toward uniform weights (feasible by convexity of g
    whenever epsilon >= JS of the uniform mixture, which the controller's
    relative calibration ensures). The KKT multiplier nu is recovered
    post-hoc by least squares on the simplex tangent space and reported for
    logging/analysis. C = clients per round, so this costs microseconds.
    """
    s = np.asarray(scores, dtype=float)
    C = s.size

    def resp(logits: np.ndarray) -> np.ndarray:
        z = (logits - logits.max()) / max(tau, 1e-8)
        e = np.exp(z)
        return e / e.sum()

    w_free = resp(s)
    if P is None or q is None or epsilon is None:
        return w_free, {"nu": 0.0, "js": float("nan"), "feasible": True,
                        "active": False, "solver_called": False,
                        "solver_success": None, "solver_message": None,
                        "fallback_used": False, "fallback_reason": None}

    js_free, _ = _js_and_grad(w_free, P, q)
    if js_free <= epsilon:
        return w_free, {"nu": 0.0, "js": float(js_free), "feasible": True,
                        "active": False, "solver_called": False,
                        "solver_success": None, "solver_message": None,
                        "fallback_used": False, "fallback_reason": None}

    from scipy.optimize import minimize

    lo = 1e-9

    def neg_obj(w):
        wc = np.clip(w, lo, None)
        return -(float(w @ s) + tau * float(-np.sum(wc * np.log(wc))))

    def neg_obj_grad(w):
        wc = np.clip(w, lo, None)
        return -(s - tau * (np.log(wc) + 1.0))

    def cons_val(w):
        js, _ = _js_and_grad(np.clip(w, lo, None), P, q)
        return epsilon - js

    def cons_grad(w):
        _, g = _js_and_grad(np.clip(w, lo, None), P, q)
        return -g

    uniform = np.ones(C) / C
    x0 = 0.5 * uniform + 0.5 * w_free
    res = minimize(
        neg_obj, x0, jac=neg_obj_grad, method="SLSQP",
        bounds=[(lo, 1.0)] * C,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(C)},
            {"type": "ineq", "fun": cons_val, "jac": cons_grad},
        ],
        options={"maxiter": 300, "ftol": 1e-12},
    )
    w = np.clip(res.x, lo, None)
    w = w / w.sum()
    js_w, grad_g = _js_and_grad(w, P, q)

    fallback_used = False
    fallback_reason = None
    if (not res.success) or js_w > epsilon * (1.0 + 1e-4):
        fallback_used = True
        fallback_reason = ("slsqp_failed" if not res.success
                           else "constraint_violated_after_solve")
        g_lo, g_hi = 0.0, 1.0
        for _ in range(60):
            gamma = 0.5 * (g_lo + g_hi)
            w_try = (1.0 - gamma) * w_free + gamma * uniform
            js_try, _ = _js_and_grad(w_try, P, q)
            if js_try <= epsilon:
                g_hi = gamma
            else:
                g_lo = gamma
        w = (1.0 - g_hi) * w_free + g_hi * uniform
        w = w / w.sum()
        js_w, grad_g = _js_and_grad(w, P, q)

    gf = s - tau * (np.log(np.clip(w, lo, None)) + 1.0)
    tf = gf - gf.mean()
    tg = grad_g - grad_g.mean()
    denom = float(tg @ tg)
    nu = float(tf @ tg) / denom if denom > 1e-18 else 0.0

    return w, {
        "nu": max(nu, 0.0),
        "js": float(js_w),
        "feasible": bool(js_w <= epsilon * (1.0 + 1e-3)),
        "active": True,
        "solver_called": True,
        "solver_success": bool(res.success),
        "solver_message": str(res.message),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


class InfoOptController:
    """InfoOpt v2: maximum-entropy client allocation under a
    distribution-preservation constraint.

    Each round the server solves
        max_w   sum_i w_i * s_i  +  tau * H(w)
        s.t.    w in simplex,
                JS( sum_i w_i P_i  ||  P_global ) <= epsilon
    where s_i aggregates (EMA-smoothed) MI, entropy and stability signals,
    P_i are client label distributions, and epsilon is calibrated per round
    relative to the uniform-weight mixture (so the problem is always feasible).

    tau is chosen adaptively as the smallest value whose solution satisfies
    an effective-client-number floor  exp(H(w)) >= kappa, which replaces the
    old free "temperature" hyperparameter with an interpretable diversity
    constraint and rules out weight collapse by construction.
    """

    def __init__(self, cfg: dict, global_label_hist: np.ndarray, variant: str = "InfoOpt-Full"):
        self.cfg = cfg["infoopt"]
        # STAGE1 PATCH: retain dataset/seed for the deterministic shuffle seed
        self._dataset_name = str(cfg.get("dataset", {}).get("name", "?"))
        self._seed = cfg.get("seed", 42)
        self.variant = variant
        self.global_label_hist = global_label_hist.astype(float) + 1e-12
        self.global_label_hist /= self.global_label_hist.sum()
        self.prev_acc = None
        self.prev_mi = None
        self.prev_entropy = None
        self.bad_rounds = 0

        self.use_mi = variant not in {"InfoOpt-NoMI"}
        self.use_entropy = variant not in {"InfoOpt-NoEntropy"}
        self.use_js = variant not in {"InfoOpt-NoJS"}
        self.use_stability = variant not in {"InfoOpt-NoStability"}
        self.use_early_stop = variant not in {"InfoOpt-NoStop"}

        self._mi_ema: dict[int, float] = {}
        self._ent_ema: dict[int, float] = {}

    def _smooth(self, store: dict[int, float], cid: int, value: float, beta: float) -> float:
        if cid not in store:
            store[cid] = value
        else:
            store[cid] = beta * store[cid] + (1.0 - beta) * value
        return store[cid]

    @staticmethod
    def _z(x: np.ndarray, robust: bool) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if robust:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            return (x - med) / (mad + 1e-8)
        return (x - x.mean()) / (x.std() + 1e-8)

    def compute_weights(self, results, round_idx=None):
        C = len(results)
        beta = float(self.cfg.get("signal_ema_beta", 0.5))

        mi = np.array(
            [self._smooth(self._mi_ema, r.client_id, r.mi_proxy, beta) for r in results],
            dtype=float,
        )
        ent = np.array(
            [self._smooth(self._ent_ema, r.client_id, r.entropy, beta) for r in results],
            dtype=float,
        )
        stability = -np.array([r.update_norm + r.train_loss_std for r in results], dtype=float)

        P = np.stack([r.label_hist.astype(float) + 1e-12 for r in results])
        P = P / P.sum(axis=1, keepdims=True)
        q = self.global_label_hist
        js = np.array([js_divergence(P[i], q) for i in range(C)], dtype=float)

        for arr, name in [(mi, "mi"), (ent, "entropy"), (js, "js"), (stability, "stability")]:
            if not np.isfinite(arr).all():
                # STAGE1: reject non-finite evidence; never quietly
                # replace bad signals with medians
                raise ValueError(f"non-finite InfoOpt signal: {name}")

        robust = bool(self.cfg.get("robust_scale", True))
        score = np.zeros(C, dtype=float)
        if self.use_mi:
            score += float(self.cfg.get("lambda_mi", 1.0)) * self._z(mi, robust)
        if self.use_entropy:
            score -= float(self.cfg.get("lambda_entropy", 0.5)) * self._z(ent, robust)
        if self.use_stability:
            score += float(self.cfg.get("lambda_stability", 0.2)) * self._z(stability, robust)

        js_mode = str(self.cfg.get("js_mode", "constraint"))
        if self.use_js and js_mode == "penalty":
            score -= float(self.cfg.get("lambda_js", 0.5)) * self._z(js, robust)

        clip = self.cfg.get("score_clip", None)
        if clip is not None:
            clip = abs(float(clip))
            score = np.clip(score, -clip, clip)

        # STAGE1 PATCH: shuffled-score control. One uniform permutation of
        # the FINAL composite score vector, AFTER EMA/scaling/clipping,
        # BEFORE the solver. Histograms/states/EMA stay in original order.
        shuffle_record = None
        if str(self.cfg.get("score_shuffle", "")) == "stage1":
            if round_idx is None:
                raise ValueError("stage1 score shuffle requires round_idx")
            seed_int = stage1_shuffle_seed(self._dataset_name,
                                           self._seed, round_idx)
            rng = np.random.default_rng(seed_int)
            perm = rng.permutation(C)
            original = score.copy()
            score = score[perm]
            shuffle_record = {
                "round": round_idx, "seed_int": seed_int,
                "permutation": [int(i) for i in perm.tolist()],
                "original_scores": [float(x) for x in original.tolist()],
                "solver_scores": [float(x) for x in score.tolist()],
            }

        if not np.isfinite(score).all():
            raise ValueError("non-finite composite score")

        constrained = self.use_js and js_mode == "constraint"
        epsilon = None
        if constrained:
            uniform = np.ones(C) / C
            js_uniform, _ = _js_and_grad(uniform, P, q)
            eps_rel = float(self.cfg.get("epsilon_rel", 1.0))
            epsilon = max(js_uniform * eps_rel, 1e-8)

        tau = max(float(self.cfg.get("temperature", 1.0)), 1e-6)
        kappa_frac = float(self.cfg.get("kappa_frac", 0.0))
        solver_calls = 0
        weights, dual_info = solve_entropic_weights(
            score, tau,
            P if constrained else None, q if constrained else None, epsilon,
        )
        solver_calls += 1
        if kappa_frac > 0.0:
            kappa = min(max(kappa_frac, 1.0 / C), 1.0) * C
            for _ in range(40):
                if np.exp(_entropy_of(weights)) >= kappa:
                    break
                tau *= 1.3
                solver_calls += 1
                weights, dual_info = solve_entropic_weights(
                    score, tau,
                    P if constrained else None, q if constrained else None, epsilon,
                )

        fedavg_mix = min(max(float(self.cfg.get("fedavg_mix", 0.0)), 0.0), 1.0)
        if fedavg_mix > 0.0:
            ns = np.array([r.num_samples for r in results], dtype=float)
            ns = ns / ns.sum() if ns.sum() > 0 else np.ones(C) / C
            weights = fedavg_mix * ns + (1.0 - fedavg_mix) * weights
            weights /= weights.sum()

        min_w = float(self.cfg.get("min_weight", 0.0))
        min_w_fraction = float(self.cfg.get("min_weight_fraction", 0.0))
        if min_w_fraction > 0.0:
            min_w = max(min_w, min_w_fraction / C)
        min_w = min(max(min_w, 0.0), (1.0 - 1e-8) / C)
        if min_w > 0.0:
            weights = np.maximum(weights, min_w)
            weights /= weights.sum()

        for r, s in zip(results, score.tolist()):
            r.score = float(s)

        mix_js, _ = _js_and_grad(weights, P, q)
        info = self._info(
            mi, ent, js, stability, score, weights,
            temp=tau, nu=dual_info["nu"],
            epsilon=(epsilon if epsilon is not None else float("nan")),
            mix_js=mix_js,
        )
        # STAGE1: explicit solver-call accounting (final call vs earlier)
        info["solver_calls"] = solver_calls
        info["solver_final_status"] = {
            "called": dual_info.get("solver_called"),
            "success": dual_info.get("solver_success"),
            "message": dual_info.get("solver_message"),
        }
        info["fallback_used"] = dual_info.get("fallback_used")
        info["fallback_reason"] = dual_info.get("fallback_reason")
        info["constraint_feasible"] = dual_info.get("feasible")
        info["constraint_active"] = dual_info.get("active")
        info["shuffle_record"] = shuffle_record
        return weights.tolist(), info

    @staticmethod
    def _info(mi, ent, js, stability, score, weights, temp, nu, epsilon, mix_js) -> dict:
        w = np.asarray(weights, dtype=float)
        return {
            "mean_mi": float(np.mean(mi)),
            "mean_entropy": float(np.mean(ent)),
            "mean_js": float(np.mean(js)),
            "mean_stability": float(np.mean(stability)),
            "mean_score": float(np.mean(score)),
            "scores": [float(s) for s in np.asarray(score).tolist()],
            "weight_var": float(np.var(w)),
            "temperature_used": float(temp),
            "nu": float(nu),
            "epsilon": float(epsilon),
            "mixture_js": float(mix_js),
            "effective_clients": float(np.exp(_entropy_of(w))),
        }

    def should_stop(self, round_idx: int, acc: float, info: dict) -> bool:
        es = self.cfg.get("early_stop", {})
        if not self.use_early_stop or not es.get("enabled", True) or round_idx < es.get("min_rounds", 20):
            self.prev_acc = acc
            self.prev_mi = info.get("mean_mi", 0.0)
            self.prev_entropy = info.get("mean_entropy", 0.0)
            return False
        acc_delta = 0.0 if self.prev_acc is None else abs(acc - self.prev_acc)
        mi_delta = 0.0 if self.prev_mi is None else abs(info.get("mean_mi", 0.0) - self.prev_mi)
        ent_delta = 0.0 if self.prev_entropy is None else abs(info.get("mean_entropy", 0.0) - self.prev_entropy)
        wvar = info.get("weight_var", 0.0)
        checks = [int(acc_delta < es.get("tau_acc", 0.001))]
        if self.use_mi:
            checks.append(int(mi_delta < es.get("tau_mi", 0.001)))
        if self.use_entropy:
            checks.append(int(ent_delta < es.get("tau_entropy", 0.001)))
        if self.use_stability:
            checks.append(int(wvar < es.get("tau_weight_var", 0.0005)))
        conditions = sum(checks)
        required = min(int(es.get("required_conditions", 2)), len(checks))
        if conditions >= required:
            self.bad_rounds += 1
        else:
            self.bad_rounds = 0
        self.prev_acc = acc
        self.prev_mi = info.get("mean_mi", 0.0)
        self.prev_entropy = info.get("mean_entropy", 0.0)
        return self.bad_rounds >= es.get("patience", 10)
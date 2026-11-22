from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from utils.feddyn import feddyn_local_penalty
from utils.info_metrics import probe_signals, state_dict_l2_delta


def _ib_terms_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=1).clamp_min(1e-12)
    mean_p = probs.mean(dim=0)
    h_mean = -(mean_p * mean_p.log()).sum()
    h_cond = -(probs * probs.log()).sum(dim=1).mean()
    mi = h_mean - h_cond
    ent = h_cond
    return mi, ent


@dataclass
class ClientResult:
    client_id: int
    state: dict
    num_samples: int
    local_steps: int
    train_loss: float
    train_loss_std: float
    entropy: float
    mi_proxy: float
    update_norm: float
    label_hist: np.ndarray
    score: float | None = None
    scaffold_ci_next: dict | None = None
    scaffold_ci_delta: dict | None = None
    local_prototypes: torch.Tensor | None = None
    local_proto_counts: torch.Tensor | None = None


def local_momentum(cfg: dict, method: str) -> float:
    if method in ("scaffold", "feddyn"):
        return 0.0
    return float(cfg["training"].get("momentum", 0.0))


def _precompute_label_hist(dataset, indices, num_classes: int) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        y = np.asarray(targets)[np.asarray(list(indices))]
        return np.bincount(y, minlength=num_classes).astype(float)
    y = np.array([int(dataset[i][1]) for i in indices])
    return np.bincount(y, minlength=num_classes).astype(float)


def train_client(
    client_id: int,
    global_model,
    dataset,
    indices,
    cfg: dict,
    device: str,
    method: str = "fedavg",
    scaffold_server_c: dict | None = None,
    scaffold_client_c: dict | None = None,
    feddyn_y: dict | None = None,
    global_prototypes: torch.Tensor | None = None,
    compute_info_signals: bool = True,
) -> ClientResult:
    model = copy.deepcopy(global_model).to(device)
    model.train()
    old_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    use_cuda = str(device).startswith("cuda")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 2)),
        pin_memory=bool(cfg["training"].get("pin_memory", use_cuda)),
    )
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=local_momentum(cfg, method),
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )
    criterion = torch.nn.CrossEntropyLoss()
    global_params = [p.detach().clone() for p in global_model.to(device).parameters()]
    fedprox_mu = float(cfg["training"].get("fedprox_mu", 0.0))
    feddyn_alpha = float(cfg["training"].get("feddyn_alpha", 0.01))
    global_state_ref = None
    if method == "feddyn" and feddyn_y is not None:
        global_state_ref = {k: v.detach() for k, v in global_model.state_dict().items()}
    ib_lambda_mi = float(cfg["training"].get("ib_lambda_mi", 0.1))
    ib_lambda_entropy = float(cfg["training"].get("ib_lambda_entropy", 0.1))
    feddp_lambda = float(cfg["training"].get("feddp_lambda", 0.1))
    clip_grad_norm = float(cfg["training"].get("clip_grad_norm", 0.0))
    is_scaffold = method == "scaffold" and scaffold_server_c is not None and scaffold_client_c is not None

    num_classes = cfg["dataset"]["num_classes"]
    label_hist = _precompute_label_hist(dataset, indices, num_classes)

    loss_sum_t = torch.zeros((), device=device)
    n = 0
    step_losses_t: list[torch.Tensor] = []
    local_steps = 0

    local_prototypes = None
    local_proto_counts = None
    feat_dim = getattr(getattr(model, "classifier", None), "in_features", 0)
    if method == "feddp" and feat_dim > 0:
        local_prototypes = torch.zeros(num_classes, feat_dim, device=device)
        local_proto_counts = torch.zeros(num_classes, device=device)

    for _ in range(cfg["training"]["local_epochs"]):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            if method == "feddp":
                logits, feat = model(x, return_features=True)
            else:
                logits = model(x)
                feat = None

            loss = criterion(logits, y)

            if method == "fedprox" and fedprox_mu > 0:
                prox = 0.0
                for p, gp in zip(model.parameters(), global_params):
                    prox = prox + torch.sum((p - gp) ** 2)
                loss = loss + 0.5 * fedprox_mu * prox

            if method == "feddyn" and feddyn_y is not None:
                loss = loss + feddyn_local_penalty(model, feddyn_y, global_state_ref, feddyn_alpha, device)

            if method == "ibfl":
                mi_t, ent_t = _ib_terms_from_logits(logits)
                loss = loss + ib_lambda_entropy * ent_t - ib_lambda_mi * mi_t

            if method == "feddp" and feat is not None and global_prototypes is not None:
                gp = global_prototypes.to(device)
                gp_y = gp[y.long()]
                valid = gp_y.abs().sum(dim=1) > 0
                if valid.any():
                    proto_loss = F.mse_loss(feat[valid], gp_y[valid])
                    loss = loss + feddp_lambda * proto_loss

            loss.backward()

            if is_scaffold:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        c = scaffold_server_c[name].to(device)
                        ci = scaffold_client_c[name].to(device)
                        p.grad.add_(c - ci)

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            opt.step()
            local_steps += 1

            bs = y.numel()
            loss_detached = loss.detach()
            loss_sum_t += loss_detached * bs
            n += bs
            step_losses_t.append(loss_detached)

            if method == "feddp" and feat is not None and local_prototypes is not None and local_proto_counts is not None:
                with torch.no_grad():
                    local_prototypes.index_add_(0, y, feat.detach())
                    local_proto_counts.index_add_(0, y, torch.ones_like(y, dtype=local_proto_counts.dtype))

    step_losses = torch.stack(step_losses_t).cpu().numpy() if step_losses_t else np.zeros(0)
    train_loss = float(loss_sum_t.cpu()) / max(n, 1)
    train_loss_std = float(step_losses.std()) if step_losses.size else 0.0

    entropy, mi_proxy = 0.0, 0.0
    if compute_info_signals:
        io_cfg = cfg.get("infoopt", {}) or {}
        entropy, mi_proxy = probe_signals(
            model,
            dataset,
            indices,
            device,
            probe_size=int(io_cfg.get("probe_size", 256)),
            batch_size=int(io_cfg.get("probe_batch_size", 256)),
            normalize_mi=bool(io_cfg.get("normalize_mi", True)),
        )

    new_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    scaffold_ci_next = None
    scaffold_ci_delta = None
    if is_scaffold:
        eta = float(cfg["training"]["lr"])
        k_steps = max(local_steps, 1)
        coeff = 1.0 / (k_steps * eta)
        scaffold_ci_next = {}
        scaffold_ci_delta = {}
        with torch.no_grad():
            for name, p_global in global_model.named_parameters():
                p_local = model.state_dict()[name].detach().to(device)
                ci_prev = scaffold_client_c[name].to(device)
                c = scaffold_server_c[name].to(device)
                ci_new = ci_prev - c + coeff * (p_global.detach().to(device) - p_local)
                scaffold_ci_next[name] = ci_new.detach().cpu().clone()
                scaffold_ci_delta[name] = (ci_new - ci_prev).detach().cpu().clone()

    if method == "feddp" and local_prototypes is not None and local_proto_counts is not None:
        den = local_proto_counts.unsqueeze(1).clamp_min(1.0)
        local_prototypes = (local_prototypes / den).cpu()
        local_proto_counts = local_proto_counts.cpu()

    return ClientResult(
        client_id=client_id,
        state=new_state,
        num_samples=len(indices),
        local_steps=local_steps,
        train_loss=train_loss,
        train_loss_std=train_loss_std,
        entropy=entropy,
        mi_proxy=mi_proxy,
        update_norm=state_dict_l2_delta(old_state, new_state),
        label_hist=label_hist,
        scaffold_ci_next=scaffold_ci_next,
        scaffold_ci_delta=scaffold_ci_delta,
        local_prototypes=local_prototypes,
        local_proto_counts=local_proto_counts,
    )
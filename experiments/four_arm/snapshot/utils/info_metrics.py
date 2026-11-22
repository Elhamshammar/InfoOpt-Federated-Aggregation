from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def predictive_entropy_from_logits(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=1).clamp_min(1e-12)
    ent = -(probs * probs.log()).sum(dim=1).mean()
    return float(ent.detach().cpu())


def mi_proxy_from_logits(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=1).clamp_min(1e-12)
    mean_p = probs.mean(dim=0)
    h_mean = -(mean_p * mean_p.log()).sum()
    h_cond = -(probs * probs.log()).sum(dim=1).mean()
    return float((h_mean - h_cond).detach().cpu())


@torch.no_grad()
def probe_signals(
    model,
    dataset,
    indices,
    device: str,
    probe_size: int = 256,
    batch_size: int = 256,
    normalize_mi: bool = True,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()

    probe_idx = sorted(indices)[: min(len(indices), probe_size)]
    loader = DataLoader(
        Subset(dataset, probe_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    probs_all = []
    labels_all = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs_all.append(F.softmax(logits, dim=1))
        labels_all.append(y)
    probs = torch.cat(probs_all).clamp_min(1e-12)
    labels = torch.cat(labels_all)

    mean_p = probs.mean(dim=0)
    h_mean = -(mean_p * mean_p.log()).sum()
    h_cond = -(probs * probs.log()).sum(dim=1).mean()
    mi = float((h_mean - h_cond).cpu())
    ent = float(h_cond.cpu())

    if normalize_mi:
        k = int(labels.unique().numel())
        denom = float(np.log(max(k, 2)))
        mi = mi / denom

    if was_training:
        model.train()
    return ent, mi


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def state_dict_l2_delta(old_state: dict, new_state: dict) -> float:
    total = 0.0
    with torch.no_grad():
        for k in old_state:
            if torch.is_floating_point(old_state[k]):
                total += torch.sum((new_state[k].detach().cpu() - old_state[k].detach().cpu()) ** 2).item()
    return float(total ** 0.5)
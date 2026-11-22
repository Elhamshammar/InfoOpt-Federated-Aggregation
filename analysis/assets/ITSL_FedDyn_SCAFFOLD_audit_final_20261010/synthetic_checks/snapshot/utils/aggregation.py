from __future__ import annotations
from collections import OrderedDict
import torch


def average_states(states: list[dict], weights: list[float] | None = None) -> OrderedDict:
    if not states:
        raise ValueError("No states supplied for aggregation")
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    total = float(sum(weights))
    weights = [w / total for w in weights]

    out = OrderedDict()
    for k in states[0].keys():
        if torch.is_floating_point(states[0][k]):
            out[k] = sum(w * s[k].detach().cpu() for w, s in zip(weights, states))
        else:
            out[k] = states[0][k].detach().cpu()
    return out


def model_num_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def state_delta(old_state: dict, new_state: dict) -> OrderedDict:
    out = OrderedDict()
    for k in old_state.keys():
        if torch.is_floating_point(old_state[k]):
            out[k] = new_state[k].detach().cpu() - old_state[k].detach().cpu()
        else:
            out[k] = old_state[k].detach().cpu()
    return out


def weighted_sum_states(states: list[dict], weights: list[float]) -> OrderedDict:
    if not states:
        raise ValueError("No states supplied")
    total = float(sum(weights))
    weights = [w / max(total, 1e-12) for w in weights]
    out = OrderedDict()
    for k in states[0].keys():
        if torch.is_floating_point(states[0][k]):
            out[k] = sum(w * s[k].detach().cpu() for w, s in zip(weights, states))
        else:
            out[k] = states[0][k].detach().cpu()
    return out


def apply_server_gradient(model, grad_state: dict, lr: float):
    with torch.no_grad():
        state = model.state_dict()
        for k in state.keys():
            if k in grad_state and torch.is_floating_point(state[k]):
                state[k].copy_(state[k].detach().cpu() - lr * grad_state[k].detach().cpu())
        model.load_state_dict(state, strict=True)

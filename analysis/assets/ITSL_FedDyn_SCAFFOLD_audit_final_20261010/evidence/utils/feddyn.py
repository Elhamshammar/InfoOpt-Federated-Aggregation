from __future__ import annotations

from collections import OrderedDict

import torch


def init_feddyn_state(model, num_clients: int, alpha: float) -> dict:
    """Create the persistent FedDyn server state: one dual per client.

    ``dual_k`` is the per-client dual variable of FedDyn (Acar et al., 2021,
    Algorithm 1). Following the official reference implementation, it stores
    ``dual_k = -\\nabla L_k(\\theta_k^{t-1})/alpha`` (initially 0).
    """
    duals = {
        cid: {
            name: torch.zeros_like(p.detach().cpu())
            for name, p in model.named_parameters()
            if torch.is_floating_point(p)
        }
        for cid in range(num_clients)
    }
    return {"alpha": float(alpha), "duals": duals}


def feddyn_local_penalty(model, dual: dict, global_state: dict, alpha: float, device: str) -> torch.Tensor:
    """Paper-faithful FedDyn local regularizer (Algorithm 1).

    The paper's local objective is

        L_k(theta) - <gradL_k(theta_k^{t-1}), theta>
                   + alpha/2 * ||theta - theta_{t-1}||^2

    whose regularizer gradient is ``-gradL_k(theta_k^{t-1}) + alpha*(theta - theta_{t-1})``.
    With ``dual_k = -gradL_k(theta_k^{t-1})/alpha`` this is exactly the gradient of

        alpha/2 * ||theta - theta_{t-1}||^2 + alpha * <theta, dual_k>

    so the returned penalty is ``alpha/2 * ||theta - theta_{t-1}||^2 + alpha * <theta, dual>``.
    This matches the official implementation whose linear term is
    ``alpha * <theta, -cld + lk>`` with ``lk = dual`` and anchor ``cld = theta_{t-1}``.
    """
    reg = torch.zeros((), device=device)
    lin = torch.zeros((), device=device)
    for name, p in model.named_parameters():
        if not torch.is_floating_point(p):
            continue
        gp = global_state[name].to(device)
        reg = reg + torch.sum((p - gp) ** 2)
        lin = lin + torch.sum(dual[name].to(device) * p)
    return 0.5 * alpha * reg + alpha * lin


def feddyn_server_update(duals: dict, results, prev_global_state: dict) -> OrderedDict:
    """Update FedDyn duals and compute the new global state.

    Dual recursion (selected clients only, matching the official reference):
        dual_k += (theta_k^t - theta_{t-1})

    which is the paper's recursion ``gradL_k(theta_k^t) = gradL_k(theta_k^{t-1})
    - alpha*(theta_k^t - theta_{t-1})`` in the ``dual_k = -gradL_k/alpha``
    parameterization.

    Server aggregation (uniform mean of selected clients + bias correction
    over ALL clients' duals, matching Algorithm 1 / the official reference):
        theta^t = (1/|P_t|) * sum_{k in P_t} theta_k^t + (1/N) * sum_k dual_k

    ``results`` entries must expose ``client_id`` and ``state`` (the local
    state theta_k^t). ``prev_global_state`` is the global state broadcast at
    the start of the round (theta_{t-1}). The ``duals`` dict is updated in
    place.
    """
    for r in results:
        cid = r.client_id
        for name in duals[cid].keys():
            delta = r.state[name].detach().cpu() - prev_global_state[name].detach().cpu()
            duals[cid][name] = duals[cid][name] + delta

    gamma = OrderedDict()
    for k in results[0].state.keys():
        if torch.is_floating_point(results[0].state[k]):
            gamma[k] = sum(r.state[k].detach().cpu() for r in results) / len(results)
        else:
            gamma[k] = results[0].state[k].detach().cpu()

    num_clients = len(duals)
    first = next(iter(duals.values()))
    dual_mean = OrderedDict()
    for name in first.keys():
        dual_mean[name] = sum(duals[cid][name] for cid in range(num_clients)) / num_clients

    new_state = OrderedDict()
    for name in gamma.keys():
        if torch.is_floating_point(gamma[name]) and name in dual_mean:
            new_state[name] = gamma[name] + dual_mean[name]
        else:
            new_state[name] = gamma[name]
    return new_state

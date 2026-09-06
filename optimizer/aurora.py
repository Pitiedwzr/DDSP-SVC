# optimizer/aurora.py
import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import List
from .chained_optimizer import ChainedOptimizer, OptimizerSpec

from .muon import get_bf16_support_map

@torch.no_grad()
def polar(G: torch.Tensor, use_bf16: bool = True) -> torch.Tensor:
    """Polar factor via 12-step simple-quintic Newton-Schulz.

    Args:
        G: input matrix of shape [..., m, n].
        use_bf16: whether to use bfloat16 (falls back to float32).

    Returns:
        polar(G) of the same shape. All non-zero singular values
        of G are mapped to 1.
    """
    assert G.ndim >= 2
    X = G.to(dtype=torch.bfloat16 if use_bf16 else torch.float32)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm <= 1 so the iteration converges to polar.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Simple-quintic coefficients: p(σ) = aσ + bσ³ + cσ⁵ with σ=1 super-attracting.
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.no_grad()
def aurora_update(W, G, momentum, eta=0.05, weight_decay=0.025, mu=0.95, nesterov=True, pp_iterations=2, pp_beta=0.5, eps=1e-7, use_bf16=True):
    """Core Aurora algorithm applied in-place."""
    if W.ndim != 2:
        raise ValueError(f"aurora expects 2D weight tensors, got shape {tuple(W.shape)}")

    # SGD-momentum (Nesterov by default).
    momentum.lerp_(G, 1 - mu)
    update = G.lerp_(momentum, mu) if nesterov else momentum.clone()

    # Aurora's leverage-uniform polar via diagonal preconditioning.
    m, n = update.size(-2), update.size(-1)
    if m == n:
        update = polar(update, use_bf16=use_bf16)
    else:
        transposed = m < n
        if transposed:
            update = update.mT
            m, n = n, m
        G32 = update.to(torch.float32)
        target_row_sq = n / m
        row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
        D = 1.0 / row_norm
        for k in range(pp_iterations):
            U = polar(D * G32, use_bf16=use_bf16)
            if k < pp_iterations - 1:
                row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
                D = D * (target_row_sq / row_sq).pow(pp_beta)
        update = U.mT if transposed else U

    # Spectral dimension scaling (Muon convention).
    update *= max(G.size(-2), G.size(-1)) ** 0.5

    # Decoupled weight decay then apply.
    W.mul_(1 - eta * weight_decay)
    W.add_(update, alpha=-eta)
    return W


class Aurora(torch.optim.Optimizer):
    """
    PyTorch Optimizer wrapper for Aurora.
    """
    def __init__(self, params, lr=5e-4, weight_decay=0.1, momentum=0.95, nesterov=True, pp_iterations=2, pp_beta=0.5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, pp_iterations=pp_iterations, pp_beta=pp_beta)
        super().__init__(params, defaults)
        self.bf16_support_map = get_bf16_support_map()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group["params"]):
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                # Flatten Conv1d/Conv2d filters to 2D matrix (out_channels, in_channels * kernel_size...)
                if p.ndim > 2:
                    p_view = p.view(p.size(0), -1)
                    g_view = g.view(g.size(0), -1)
                    mom_view = state["momentum_buffer"].view(g.size(0), -1)
                else:
                    p_view = p
                    g_view = g
                    mom_view = state["momentum_buffer"]

                use_bf16 = self.bf16_support_map.get(g.device, False)
                aurora_update(
                    W=p_view,
                    G=g_view,
                    momentum=mom_view,
                    eta=group["lr"],
                    weight_decay=group["weight_decay"],
                    mu=group["momentum"],
                    nesterov=group["nesterov"],
                    pp_iterations=group["pp_iterations"],
                    pp_beta=group["pp_beta"],
                    use_bf16=use_bf16
                )
        return loss


def get_params_for_aurora(model) -> List[Parameter]:
    """
    Filter parameters: Use Aurora for dense weights, but exclude embeddings
    and depthwise convolutions (which should go to AdamW).
    """
    aurora_params = []
    for module in model.modules():
        # Exclude embeddings entirely
        if isinstance(module, nn.Embedding):
            continue

        # Exclude depthwise convolutions (where groups == in_channels)
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if module.groups == module.in_channels and module.in_channels > 1:
                continue

        for param in module.parameters(recurse=False):
            if not param.requires_grad:
                continue
            # Only optimize >= 2D matrices (ignoring 1D biases)
            if param.ndim >= 2:
                aurora_params.append(param)
    return aurora_params


class Aurora_AdamW(ChainedOptimizer):
    def __init__(self, model, lr=0.0005, weight_decay=0.0, aurora_args={}, adamw_args={}, verbose=False):
        aurora_params_id_set = set(id(p) for p in get_params_for_aurora(model))
        spec_aurora = OptimizerSpec(Aurora, aurora_args, lambda param: id(param) in aurora_params_id_set)
        spec_adamw = OptimizerSpec(torch.optim.AdamW, adamw_args, None)
        specs = [spec_aurora, spec_adamw]

        callback = None
        if verbose:
            callback = lambda p, spec_idx: print(
                f"Adding param {p.shape} to optimizer{spec_idx} {str(specs[spec_idx].class_type)}"
            )
        super().__init__(model.parameters(), specs, lr=lr, weight_decay=weight_decay, optimizer_selection_callback=callback)
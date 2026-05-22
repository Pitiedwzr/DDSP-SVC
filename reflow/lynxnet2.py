import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm1d(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

class Linear1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + 'weight'
        if weight_key in state_dict and state_dict[weight_key].ndim == 2:
            state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class SwiGLU(nn.Module):
    # Swish-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        gate = F.silu(gate)
        return out * gate


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor=2, dim_global_cond=256, kernel_size=31, dilation=1, dropout=0.):
        super().__init__()

        self.norm = LayerNorm1d(dim)

        # FiLM -> AdaLN-Zero
        self.film_proj = nn.Linear(dim_global_cond * 2, dim * 3)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        # Spatial Gating Projections
        self.proj_v = Linear1d(dim, dim)
        self.proj_gate = Linear1d(dim, dim)

        # Dilation
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim)

        # Single Clean MLP
        inner_dim = int(dim * expansion_factor)
        self.mlp = nn.Sequential(
            Linear1d(dim, inner_dim * 2),
            SwiGLU(dim=1),
            Linear1d(inner_dim, dim),
            nn.Dropout(dropout) if float(dropout) > 0. else nn.Identity()
        )

    def forward(self, x, global_cond):
        res = x
        x = self.norm(x)

        # FiLM -> AdaLN-Zero
        film_params = self.film_proj(global_cond).unsqueeze(-1) # [B, 3*dim, 1]
        gamma, beta, alpha = film_params.chunk(3, dim=1) # [B, dim, 1]

        x = x * (1 + gamma) + beta

        # Spatial Gating
        v = self.proj_v(x)
        gate = self.conv(x)
        gate = self.proj_gate(gate)

        x = v * torch.atan(gate)

        x = self.mlp(x)

        # Because alpha is initialized to 0, this block starts as a pure Identity function.
        return res + x * alpha


class LYNXNet2(nn.Module):
    def __init__(self, in_dims, dim_cond, dim_global_cond=256, n_layers=6, n_chans=512, expansion_factor=2, dropout=0.):
        """
        LYNXNet2(Linear Gated Depthwise Separable Convolution Network Version 2)
        """
        super().__init__()
        self.input_projection = Linear1d(in_dims, n_chans)
        self.conditioner_projection = Linear1d(dim_cond, n_chans)

        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, dim_global_cond),
        )

        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=n_chans,
                    expansion_factor=expansion_factor,
                    dim_global_cond=dim_global_cond,
                    kernel_size=31,
                    dilation=2 ** (i % 4), # 1, 2, 4, 8 loop
                    dropout=dropout
                )
                for i in range(n_layers)
            ]
        )
        self.norm = LayerNorm1d(n_chans)
        self.output_projection = Linear1d(n_chans, in_dims)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond, global_cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """

        # To keep compatibility with DiffSVC, [B, 1, M, T]
        x = spec
        use_4_dim = False
        if x.dim() == 4:
            x = x[:, 0]
            use_4_dim = True

        assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"

        x = self.input_projection(x)
        x = x + self.conditioner_projection(cond)

        time_emb = self.diffusion_embedding(diffusion_step)        # [B, dim_global_cond]
        block_cond = torch.cat([global_cond, time_emb], dim=-1)    # [B, dim_global_cond]

        for layer in self.residual_layers:
            x = layer(x, block_cond)

        # post-norm
        x = self.norm(x)

        # output projection
        x = self.output_projection(x)  # [B, 128, T]

        return x[:, None] if use_4_dim else x
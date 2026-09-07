"""
LYNXNet2-AdaLN (Modified LYNXNet2 Backbone)

This is a modified version of the original LYNXNet2 backbone.
It combines the base structure of LYNXNet2 with the Spatial Gating from LYNXNet2Plus,
alongside custom modifications for AdaLN-Zero conditioning, also some other modifications.

Acknowledgements:

- Original LYNXNet2 by yxlllc: https://github.com/yxlllc/DDSP-SVC: Provided the base architecture
- LYNXNet2Plus by KakaruHayate: https://github.com/KakaruHayate/DiffSinger/tree/lynxnet2attn: Inspired the integration of Spatial Gating (atan gating).

Key Modifications:

- Replaced standard diffusion embedding addition with AdaLN-Zero conditioning.
- Integrated Spatial Gating Projections from LYNXNet2Plus.
- Added cyclical dilation (1, 2, 4, 8) to the convolutional layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Accept both utterance timesteps [B] and token timesteps [B, T].
        emb = x[..., None] * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LYNXNet2AdaLNBlock(nn.Module):
    def __init__(self, dim, expansion_factor=2, dim_global_cond=256, kernel_size=31, dilation=1, dropout=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        # FiLM -> AdaLN-Zero
        self.film_proj = nn.Linear(dim_global_cond * 2, dim * 3)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        # Spatial Gating Projections
        self.proj_v = nn.Linear(dim, dim)
        self.proj_gate = nn.Linear(dim, dim)

        # Dilation
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim)

        # Single Clean MLP
        inner_dim = int(dim * expansion_factor)
        self.mlp = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout) if float(dropout) > 0. else nn.Identity()
        )

    def forward(self, x, global_cond):
        res = x
        x = self.norm(x)

        # FiLM -> AdaLN-Zero
        film_params = self.film_proj(global_cond)
        if film_params.dim() == 2:
            film_params = film_params.unsqueeze(1)
        gamma, beta, alpha = film_params.chunk(3, dim=-1)

        x = x * (1 + gamma) + beta

        # Spatial Gating
        v = self.proj_v(x)
        gate = x.transpose(1, 2)
        gate = self.conv(gate)
        gate = gate.transpose(1, 2)
        gate = self.proj_gate(gate)

        x = v * torch.atan(gate)

        x = self.mlp(x)

        # Because alpha is initialized to 0, this block starts as a pure Identity function.
        return res + x * alpha


class LYNXNet2AdaLN(nn.Module):
    def __init__(self, in_dims, dim_cond, dim_global_cond=256, n_layers=6, n_chans=512,
                 expansion_factor=2, dropout=0., use_self_flow=False,
                 self_flow_projector_dim=1024):
        """
        LYNXNet2(Linear Gated Depthwise Separable Convolution Network Version 2)
        """
        super().__init__()
        self.input_projection = nn.Linear(in_dims, n_chans)
        self.conditioner_projection = nn.Linear(dim_cond, n_chans)

        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, dim_global_cond),
        )

        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2AdaLNBlock(
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
        self.norm = nn.LayerNorm(n_chans)
        self.output_projection = nn.Linear(n_chans, in_dims)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        self.self_flow_projector = None
        if use_self_flow:
            if not 1 <= self_flow_projector_dim:
                raise ValueError("self_flow_projector_dim must be positive")
            self.self_flow_projector = nn.Sequential(
                nn.Linear(n_chans, self_flow_projector_dim),
                nn.SiLU(),
                nn.Linear(self_flow_projector_dim, n_chans)
            )

    def forward(self, spec, diffusion_step, cond, global_cond, return_hidden_layer=None):
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

        x = self.input_projection(x.transpose(1, 2))
        x = x + self.conditioner_projection(cond.transpose(1, 2))

        time_emb = self.diffusion_embedding(diffusion_step)
        if time_emb.dim() == 3 and time_emb.size(1) == 1:
            time_emb = time_emb.squeeze(1)
        if global_cond.dim() == 3 and global_cond.size(1) == 1:
            global_cond = global_cond.squeeze(1)

        if time_emb.dim() == 3:
            if global_cond.dim() == 2:
                global_cond = global_cond.unsqueeze(1).expand(-1, x.size(1), -1)
            elif global_cond.size(1) == 1:
                global_cond = global_cond.expand(-1, x.size(1), -1)
        elif global_cond.dim() == 3:
            time_emb = time_emb.unsqueeze(1).expand(-1, x.size(1), -1)

        block_cond = torch.cat([global_cond, time_emb], dim=-1)

        selected_hidden = None
        for layer_index, layer in enumerate(self.residual_layers, start=1):
            x = layer(x, block_cond)
            if layer_index == return_hidden_layer:
                selected_hidden = x

        # post-norm
        x = self.norm(x)

        # output projection
        x = self.output_projection(x).transpose(1, 2)  # [B, 128, T]

        output = x[:, None] if use_4_dim else x
        if return_hidden_layer is not None:
            if selected_hidden is None:
                raise ValueError(
                    f"return_hidden_layer must be between 1 and {len(self.residual_layers)}, "
                    f"got {return_hidden_layer}")
            return output, selected_hidden
        return output

    def project_self_flow(self, hidden):
        if self.self_flow_projector is None:
            raise RuntimeError("Self-Flow projector is not enabled")
        return self.self_flow_projector(hidden)

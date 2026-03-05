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
        if x.dtype == torch.float16:
            out_min, out_max = torch.aminmax(out.detach())
            gate_min, gate_max = torch.aminmax(gate.detach())
            max_abs_out = torch.max(-out_min, out_max).float()
            max_abs_gate = torch.max(-gate_min, gate_max).float()
            max_abs_value = max_abs_out * max_abs_gate
            if max_abs_value > 1000:
                ratio = (1000 / max_abs_value).half()
                gate *= ratio
                return (out * gate).clamp(-1000 * ratio, 1000 * ratio) / ratio
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


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor=2, dim_global_cond=256, kernel_size=31, dilation=1, dropout=0.):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        
        self.norm = nn.LayerNorm(dim)
        
        # FiLM
        self.film_proj = nn.Linear(dim_global_cond * 2, dim * 2)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        
        # Dilation
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim)
        
        # SwiGLU * 1 with 2 expansion_factor
        self.ffn = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout) if float(dropout) > 0. else nn.Identity()
        )

    def forward(self, x, global_cond):
        res = x
        x = self.norm(x)
        
        # FiLM
        film_params = self.film_proj(global_cond).unsqueeze(1) # [B, 1, 2*dim]
        gamma, beta = film_params.chunk(2, dim=-1) # [B, 1, dim]
        x = x * (1 + gamma) + beta
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        x = self.ffn(x)
        return res + x


class LYNXNet2(nn.Module):
    def __init__(self, in_dims, dim_cond, dim_global_cond=256, n_layers=6, n_chans=512, expansion_factor=2, dropout=0.):
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
        self.norm = nn.LayerNorm(n_chans)
        self.output_projection = nn.Linear(n_chans, in_dims)
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

        x = self.input_projection(x.transpose(1, 2))
        x = x + self.conditioner_projection(cond.transpose(1, 2))
        
        time_emb = self.diffusion_embedding(diffusion_step)        # [B, dim_global_cond]
        block_cond = torch.cat([global_cond, time_emb], dim=-1)    # [B, dim_global_cond]
        
        for layer in self.residual_layers:
            x = layer(x, block_cond)

        # post-norm
        x = self.norm(x)
        
        # output projection
        x = self.output_projection(x).transpose(1, 2)  # [B, 128, T]
        
        return x[:, None] if use_4_dim else x
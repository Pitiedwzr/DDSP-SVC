import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class RectifiedFlow(nn.Module):
    def __init__(self, 
                velocity_fn, 
                out_dims=128,
                spec_min=-12, 
                spec_max=2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max
    
    def reflow_loss(self, x_1, t, cond, global_cond=None, loss_type='l2'):
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        
        v_pred = self.velocity_fn(x_t, 1000 * t, cond, global_cond)
        
        if loss_type == 'l1':
            loss = (x_1 - x_0 - v_pred).abs().mean()
        else:
            loss = F.mse_loss(x_1 - x_0, v_pred)
        return loss

    # Add CFG
    def sample_euler(self, x, t, dt, cond, global_cond=None, cfg_scale=1.0, null_global_cond=None): 
        v_cond = self.velocity_fn(x, 1000 * t, cond, global_cond)
        
        if cfg_scale > 1.0 and null_global_cond is not None:
            null_cond = torch.zeros_like(cond)
            v_uncond = self.velocity_fn(x, 1000 * t, null_cond, null_global_cond)
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_pred = v_cond
            
        x += v_pred * dt
        t = t + dt
        return x, t

    def sample_rk2(self, x, t, dt, cond, global_cond=None, cfg_scale=1.0, null_global_cond=None):
        """
        2nd-Order Runge-Kutta (Midpoint) Sampler.
        Twice as fast as RK4, much higher quality than Euler.
        """
        def get_v(vx, vt):
            v_cond = self.velocity_fn(vx, 1000 * vt, cond, global_cond)
            if cfg_scale > 1.0 and null_global_cond is not None:
                null_cond = torch.zeros_like(cond)
                v_uncond = self.velocity_fn(vx, 1000 * vt, null_cond, null_global_cond)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            return v_cond

        # Step 1: Calculate velocity at current point
        v_1 = get_v(x, t)
        
        # Step 2: Step HALFWAY forward and calculate velocity again
        x_half = x + 0.5 * v_1 * dt
        t_half = t + 0.5 * dt
        v_2 = get_v(x_half, t_half)
        
        # Step 3: Take the full step using the halfway velocity
        x += v_2 * dt
        t = t + dt
        
        return x, t

    def sample_rk4(self, x, t, dt, cond, global_cond=None, cfg_scale=1.0, null_global_cond=None):
        def get_v(vx, vt):
            v_cond = self.velocity_fn(vx, 1000 * vt, cond, global_cond)
            if cfg_scale > 1.0 and null_global_cond is not None:
                null_cond = torch.zeros_like(cond)
                v_uncond = self.velocity_fn(vx, 1000 * vt, null_cond, null_global_cond)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            return v_cond

        k_1 = get_v(x, t)
        k_2 = get_v(x + 0.5 * k_1 * dt, t + 0.5 * dt)
        k_3 = get_v(x + 0.5 * k_2 * dt, t + 0.5 * dt)
        k_4 = get_v(x + k_3 * dt, t + dt)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t = t + dt
        return x, t

    def forward(self, 
                condition, 
                gt_spec=None, 
                global_cond=None,
                infer=True,
                infer_step=10,
                method='euler',
                t_start=0.0,
                use_tqdm=True, 
                cfg_scale=1.0, 
                null_global_cond=None):
        cond = condition.transpose(1, 2) # [B, H, T]
        b, device = condition.shape[0], condition.device
        if t_start < 0.0:
            t_start = 0.0
        if not infer:
            x_1 = self.norm_spec(gt_spec).transpose(1, 2)[:, None, :, :]
            
            # Logit-Normal From RIFT-SVC
            quantiles = torch.linspace(0, 1, b + 1, device=device)
            z_uniform = quantiles[:-1] + torch.rand((b,), device=device) / b
            z_normal = torch.erfinv(2 * z_uniform - 1) * math.sqrt(2)
            t = torch.sigmoid(z_normal)
            
            if t_start > 0.0:
                t = t_start + (1.0 - t_start) * t
            t = torch.clip(t, 1e-7, 1-1e-7)
            
            return self.reflow_loss(x_1, t, cond=cond, global_cond=global_cond)
        else:
            shape = (cond.shape[0], 1, self.out_dims, cond.shape[2]) # [B, 1, M, T]
            
            # initial condition and step size of the ODE
            if gt_spec is None:
                x = torch.randn(shape, device=device)
                t = torch.full((b,), 0, device=device)
                dt = 1.0 / infer_step 
            else:
                norm_spec = self.norm_spec(gt_spec)
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :] # [B, 1, M, T]
                x = t_start * norm_spec + (1 - t_start) * torch.randn(shape, device=device)
                t = torch.full((b,), t_start, device=device)
                dt = (1.0 - t_start) / infer_step 
                  
            if method == 'euler':
                iterator = tqdm(range(infer_step), desc='sample time step') if use_tqdm else range(infer_step)
                for i in iterator:
                    x, t = self.sample_euler(x, t, dt, cond, global_cond, cfg_scale, null_global_cond)
            elif method == 'rk2':
                iterator = tqdm(range(infer_step), desc='sample time step') if use_tqdm else range(infer_step)
                for i in iterator:
                    x, t = self.sample_rk2(x, t, dt, cond, global_cond, cfg_scale, null_global_cond)
            elif method == 'rk4':
                iterator = tqdm(range(infer_step), desc='sample time step', total=infer_step) if use_tqdm else range(infer_step)
                for i in iterator:
                    x, t = self.sample_rk4(x, t, dt, cond, global_cond, cfg_scale, null_global_cond)
            
            else:
                raise NotImplementedError(method)
                
            x = x.squeeze(1).transpose(1, 2)  # [B, T, M]
            
            return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

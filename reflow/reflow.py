import math

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .inference_utils import validate_infer_step


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        velocity_fn,
        out_dims=128,
        spec_min=-12,
        spec_max=2,
        use_self_flow=False,
        self_flow_student_layer=2,
        self_flow_teacher_layer=4,
        self_flow_mask_ratio=0.5,
        self_flow_condition_mask_ratio=0.0,
        self_flow_span_length=1,
        self_flow_loss_on_masked_only=False,
    ):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max
        self.use_self_flow = use_self_flow
        self.self_flow_student_layer = self_flow_student_layer
        self.self_flow_teacher_layer = self_flow_teacher_layer
        self.self_flow_mask_ratio = self_flow_mask_ratio
        self.self_flow_condition_mask_ratio = self_flow_condition_mask_ratio
        self.self_flow_span_length = max(1, int(self_flow_span_length))
        self.self_flow_loss_on_masked_only = self_flow_loss_on_masked_only
        self.condition_mask_token = None
        if use_self_flow:
            self.condition_mask_token = nn.Parameter(
                torch.full((1, out_dims, 1), float(spec_min))
            )

        if not 0.0 <= self.self_flow_mask_ratio <= 0.5:
            raise ValueError("self_flow_mask_ratio must be between 0 and 0.5")
        if not 0.0 <= self.self_flow_condition_mask_ratio <= 1.0:
            raise ValueError("self_flow_condition_mask_ratio must be between 0 and 1")

    def _mask_condition(self, cond):
        if self.self_flow_condition_mask_ratio <= 0.0:
            return cond
        keep = (
            torch.rand(cond.size(0), 1, cond.size(2), device=cond.device)
            >= self.self_flow_condition_mask_ratio
        )
        mask_token = self.condition_mask_token.to(dtype=cond.dtype)
        return torch.where(keep, cond, mask_token)

    def _sample_self_flow_mask(self, batch_size, seq_len, device):
        if self.self_flow_span_length <= 1:
            return (
                torch.rand(batch_size, seq_len, device=device)
                < self.self_flow_mask_ratio
            )
        p_start = min(1.0, self.self_flow_mask_ratio / self.self_flow_span_length)
        starts = (torch.rand(batch_size, 1, seq_len, device=device) < p_start).float()
        kernel = torch.ones(1, 1, self.self_flow_span_length, device=device)
        spans = F.conv1d(starts, kernel, padding=self.self_flow_span_length - 1)[
            :, 0, :seq_len
        ]
        return spans > 0

    def reflow_loss(
        self,
        x_1,
        t,
        cond,
        global_cond=None,
        loss_type="l2",
        teacher_velocity_fn=None,
        return_self_flow_loss=False,
        t_start=0.0,
    ):
        x_0 = torch.randn_like(x_1)
        target_velocity = x_1 - x_0

        self_flow_loss = x_1.sum() * 0.0
        if self.use_self_flow:
            second_t = self._sample_timesteps(x_1.size(0), x_1.device, t_start=t_start)
            if t.dim() != 1:
                raise ValueError("Base Self-Flow timesteps must have shape [B]")
            token_mask = self._sample_self_flow_mask(
                x_1.size(0), x_1.size(-1), device=x_1.device
            )
            token_t = torch.where(token_mask, second_t[:, None], t[:, None])
            x_t = x_0 + token_t[:, None, None, :] * target_velocity

            student_cond = self._mask_condition(cond)
            if teacher_velocity_fn is not None:
                v_pred, student_hidden = self.velocity_fn(
                    x_t,
                    1000 * token_t,
                    student_cond,
                    global_cond,
                    return_hidden_layer=self.self_flow_student_layer,
                )
                # This repository uses t=0 for noise and t=1 for clean data,
                # opposite to the paper. Therefore max(t, s) is the cleaner view.
                clean_t = torch.maximum(t, second_t)
                x_clean = x_0 + clean_t[:, None, None, None] * target_velocity
                with torch.no_grad():
                    _, teacher_hidden = teacher_velocity_fn(
                        x_clean,
                        1000 * clean_t,
                        cond,
                        global_cond,
                        return_hidden_layer=self.self_flow_teacher_layer,
                    )

                # The projection head and cosine reduction are especially prone
                # to fp16 overflow. Casting after the projection is too late: an
                # fp16 linear can already have produced Inf, and Inf / Inf below
                # becomes NaN. Keep only this small auxiliary path in float32.
                with torch.autocast(
                    device_type=student_hidden.device.type, enabled=False
                ):
                    student_f = self.velocity_fn.project_self_flow(
                        student_hidden.float()
                    )
                    teacher_f = teacher_hidden.float()

                    student_norm = student_f / student_f.norm(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-4)
                    teacher_norm = teacher_f / teacher_f.norm(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-4)
                    cos_sim = (student_norm * teacher_norm).sum(dim=-1)
                    cos_dist = 1.0 - cos_sim

                if self.self_flow_loss_on_masked_only and token_mask.any():
                    self_flow_loss = cos_dist[token_mask].mean()
                else:
                    self_flow_loss = cos_dist.mean()
            else:
                v_pred = self.velocity_fn(
                    x_t, 1000 * token_t, student_cond, global_cond
                )
        else:
            x_t = x_0 + t[:, None, None, None] * target_velocity
            v_pred = self.velocity_fn(x_t, 1000 * t, cond, global_cond)

        if loss_type == "l1":
            loss = (target_velocity - v_pred).abs().mean()
        else:
            loss = F.mse_loss(target_velocity, v_pred)
        return (loss, self_flow_loss) if return_self_flow_loss else loss

    def _sample_timesteps(self, batch_size, device, t_start=0.0):
        # Stratified logit-normal sampling, matching the existing training path.
        quantiles = torch.linspace(0, 1, batch_size + 1, device=device)
        z_uniform = (
            quantiles[:-1] + torch.rand((batch_size,), device=device) / batch_size
        )
        z_uniform = z_uniform[torch.randperm(batch_size, device=device)]
        z_normal = torch.erfinv(2 * z_uniform - 1) * math.sqrt(2)
        t = torch.sigmoid(z_normal)
        if t_start > 0.0:
            t = t_start + (1.0 - t_start) * t
        return torch.clip(t, 1e-7, 1 - 1e-7)

    def _get_velocity(
        self, vx, vt, cond, global_cond=None, cfg_scale=1.0, null_global_cond=None
    ):
        if cfg_scale > 1.0 and null_global_cond is not None:
            vx_cat = torch.cat([vx, vx], dim=0)
            vt_cat = torch.cat([vt, vt], dim=0)
            cond_cat = torch.cat([cond, cond], dim=0)
            gcond_cat = torch.cat([global_cond, null_global_cond], dim=0)
            v_all = self.velocity_fn(vx_cat, 1000 * vt_cat, cond_cat, gcond_cat)
            b = vx.size(0)
            v_cond = v_all[:b]
            v_uncond = v_all[b:]
            return v_uncond + cfg_scale * (v_cond - v_uncond)
        return self.velocity_fn(vx, 1000 * vt, cond, global_cond)

    # Add CFG
    def sample_euler(
        self, x, t, dt, cond, global_cond=None, cfg_scale=1.0, null_global_cond=None
    ):
        v_pred = self._get_velocity(
            x, t, cond, global_cond, cfg_scale, null_global_cond
        )
        x += v_pred * dt
        t = t + dt
        return x, t

    def sample_rk2(
        self, x, t, dt, cond, global_cond=None, cfg_scale=1.0, null_global_cond=None
    ):
        """
        2nd-Order Runge-Kutta (Midpoint) Sampler.
        Twice as fast as RK4, much higher quality than Euler.
        """
        v_1 = self._get_velocity(x, t, cond, global_cond, cfg_scale, null_global_cond)
        x_half = x + 0.5 * v_1 * dt
        t_half = t + 0.5 * dt
        v_2 = self._get_velocity(
            x_half, t_half, cond, global_cond, cfg_scale, null_global_cond
        )
        x += v_2 * dt
        t = t + dt
        return x, t

    def sample_rk4(
        self, x, t, dt, cond, global_cond=None, cfg_scale=1.0, null_global_cond=None
    ):
        k_1 = self._get_velocity(x, t, cond, global_cond, cfg_scale, null_global_cond)
        k_2 = self._get_velocity(
            x + 0.5 * k_1 * dt,
            t + 0.5 * dt,
            cond,
            global_cond,
            cfg_scale,
            null_global_cond,
        )
        k_3 = self._get_velocity(
            x + 0.5 * k_2 * dt,
            t + 0.5 * dt,
            cond,
            global_cond,
            cfg_scale,
            null_global_cond,
        )
        k_4 = self._get_velocity(
            x + k_3 * dt, t + dt, cond, global_cond, cfg_scale, null_global_cond
        )
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t = t + dt
        return x, t

    def forward(
        self,
        condition,
        gt_spec=None,
        global_cond=None,
        infer=True,
        infer_step=10,
        method="euler",
        t_start=0.0,
        use_tqdm=True,
        cfg_scale=1.0,
        null_global_cond=None,
        teacher_velocity_fn=None,
        return_self_flow_loss=False,
    ):
        cond = condition.transpose(1, 2)  # [B, H, T]
        b, device = condition.shape[0], condition.device
        t_start = max(t_start, 0.0)
        if not infer:
            x_1 = self.norm_spec(gt_spec).transpose(1, 2)[:, None, :, :]

            t = self._sample_timesteps(b, device, t_start=t_start)

            return self.reflow_loss(
                x_1,
                t,
                cond=cond,
                global_cond=global_cond,
                teacher_velocity_fn=teacher_velocity_fn,
                return_self_flow_loss=return_self_flow_loss,
                t_start=t_start,
            )
        else:
            validate_infer_step(infer_step)
            shape = (cond.shape[0], 1, self.out_dims, cond.shape[2])  # [B, 1, M, T]

            # initial condition and step size of the ODE
            if t_start <= 0.0:
                x = torch.randn(shape, device=device)
                t = torch.zeros((b,), device=device)
                dt = 1.0 / infer_step
            else:
                # Shallow refinement starts from a noised acoustic prior. In the
                # normal VC path this is the DDSP mel supplied as `condition`.
                initial_spec = condition if gt_spec is None else gt_spec
                norm_spec = self.norm_spec(initial_spec)
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
                x = t_start * norm_spec + (1 - t_start) * torch.randn(
                    shape, device=device
                )
                t = torch.full((b,), t_start, device=device)
                dt = (1.0 - t_start) / infer_step

            if method == "euler":
                iterator = (
                    tqdm(range(infer_step), desc="sample time step")
                    if use_tqdm
                    else range(infer_step)
                )
                for i in iterator:
                    x, t = self.sample_euler(
                        x, t, dt, cond, global_cond, cfg_scale, null_global_cond
                    )
            elif method == "rk2":
                iterator = (
                    tqdm(range(infer_step), desc="sample time step")
                    if use_tqdm
                    else range(infer_step)
                )
                for i in iterator:
                    x, t = self.sample_rk2(
                        x, t, dt, cond, global_cond, cfg_scale, null_global_cond
                    )
            elif method == "rk4":
                iterator = (
                    tqdm(range(infer_step), desc="sample time step", total=infer_step)
                    if use_tqdm
                    else range(infer_step)
                )
                for i in iterator:
                    x, t = self.sample_rk4(
                        x, t, dt, cond, global_cond, cfg_scale, null_global_cond
                    )

            else:
                raise NotImplementedError(method)

            x = x.squeeze(1).transpose(1, 2)  # [B, T, M]

            return self.denorm_spec(x)

    def norm_spec(self, x):
        x = torch.clamp(x, self.spec_min, self.spec_max)
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        x = torch.clamp(x, -1.0, 1.0)
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

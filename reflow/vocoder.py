import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nsf_hifigan.nvSTFT import STFT
from nsf_hifigan.models import load_model,load_config
from torchaudio.transforms import Resample
from logger import utils
from .reflow import RectifiedFlow
from .lynxnet2adaln import LYNXNet2AdaLN
from ddsp.vocoder import CombSubSuperFast

class DotDict(dict):
    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_model_vocoder(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)

    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)

    # load model
    if args.model.type == 'RectifiedFlow':
        model = Unit2Wav(
            args.data.sampling_rate,
            args.data.block_size,
            args.model.win_length,
            args.data.encoder_out_channels,
            args.model.n_spk,
            args.model.use_norm,
            args.model.use_attention,
            args.model.use_pitch_aug,
            vocoder.dimension,
            args.model.n_aux_layers,
            args.model.n_aux_chans,
            args.model.n_layers,
            args.model.n_chans,
            args.model.get('spec_min', -12),
            args.model.get('spec_max', 2),
            args.model.get('use_aux_f0', True),
            args.model.get('use_f0_conditioning', False),
            args.model.get('use_self_flow', False),
            args.model.get('self_flow_student_layer', 2),
            args.model.get('self_flow_teacher_layer', 4),
            args.model.get('self_flow_projector_dim', 1024),
            args.model.get('self_flow_mask_ratio', 0.5),
            args.model.get('self_flow_condition_mask_ratio', 0.0))

    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")

    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.to(device)
    if 'ema_model' in ckpt:
        print(' [*] Loading EMA weights...')
        ema_dict = utils.unwrap_ema_state_dict(ckpt['ema_model'])
        model.load_state_dict(ema_dict)
    else:
        print(' [*] Loading standard weights (no EMA found)')
        model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocoder, args


class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if vocoder_type == 'nsf-hifigan':
            self.vocoder = NsfHifiGAN(vocoder_ckpt, device = device)
        elif vocoder_type == 'nsf-hifigan-log10':
            self.vocoder = NsfHifiGANLog10(vocoder_ckpt, device = device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")

        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()

    def extract(self, audio, sample_rate=0, keyshift=0):

        # resample
        if sample_rate == self.vocoder_sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)

            # extract
        mel = self.vocoder.extract(audio_res, keyshift=keyshift) # B, n_frames, bins
        return mel

    def infer(self, mel, f0):
        f0 = f0[:,:mel.size(1),0] # B, n_frames
        audio = self.vocoder(mel, f0)
        return audio


class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)
        self.stft = STFT(
            self.h.sampling_rate,
            self.h.num_mels,
            self.h.n_fft,
            self.h.win_size,
            self.h.hop_size,
            self.h.fmin,
            self.h.fmax)

    def sample_rate(self):
        return self.h.sampling_rate

    def hop_size(self):
        return self.h.hop_size

    def dimension(self):
        return self.h.num_mels

    def extract(self, audio, keyshift=0):
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2) # B, n_frames, bins
        return mel

    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class NsfHifiGANLog10(NsfHifiGAN):
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class Unit2Wav(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            n_spk,
            use_norm=False,
            use_attention=False,
            use_pitch_aug=False,
            out_dims=128,
            n_aux_layers=3,
            n_aux_chans=256,
            n_layers=6,
            n_chans=512,
            spec_min=-12,
            spec_max=2,
            use_aux_f0=True,
            use_f0_conditioning=False,
            use_self_flow=False,
            self_flow_student_layer=2,
            self_flow_teacher_layer=4,
            self_flow_projector_dim=1024,
            self_flow_mask_ratio=0.5,
            self_flow_condition_mask_ratio=0.0):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size

        if use_self_flow:
            if not 1 <= self_flow_student_layer < self_flow_teacher_layer <= n_layers:
                raise ValueError(
                    "Self-Flow layers must satisfy 1 <= student_layer < "
                    f"teacher_layer <= n_layers ({n_layers})")

        self.spk_embed = nn.Embedding(n_spk, 256)
        self.unit_hidden_dim = n_unit
        self.shared_unit_encoder = nn.Sequential(
            nn.Conv1d(n_unit, self.unit_hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(self.unit_hidden_dim, self.unit_hidden_dim, kernel_size=3, padding=1)
        )
        self.f0_predictor = nn.Sequential(
            nn.Conv1d(self.unit_hidden_dim + 256, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(128, 1, 1) # [B, 1, T]
        ) if use_aux_f0 else None

        self.ddsp_model = CombSubSuperFast(
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            n_spk,
            n_aux_layers if n_aux_layers is not None else 3,
            n_aux_chans if n_aux_chans is not None else 256,
            use_norm,
            use_attention,
            use_pitch_aug,
            use_f0_conditioning)
        self.reflow_model = RectifiedFlow(
            LYNXNet2AdaLN(
                in_dims=out_dims,
                dim_cond=out_dims,
                dim_global_cond=256,           # (Spk Emb 维度)
                n_layers=n_layers,
                n_chans=n_chans,
                use_self_flow=use_self_flow,
                self_flow_projector_dim=self_flow_projector_dim
            ),
            out_dims=out_dims,
            spec_min=spec_min,
            spec_max=spec_max,
            use_self_flow=use_self_flow,
            self_flow_student_layer=self_flow_student_layer,
            self_flow_teacher_layer=self_flow_teacher_layer,
            self_flow_mask_ratio=self_flow_mask_ratio,
            self_flow_condition_mask_ratio=self_flow_condition_mask_ratio
        )

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0,
                silence_front=0, use_tqdm=True, cfg_scale=1.0, drop_spk=False,
                teacher_velocity_fn=None, return_self_flow_loss=False):

        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        # 1. Calculate the TRUE speaker embedding
        if spk_mix_dict is not None:
            true_spk_emb = torch.zeros((units.shape[0], 256), device=units.device)
            for k, v in spk_mix_dict.items():
                mix_id = torch.tensor([[int(k)]], dtype=torch.long, device=units.device)
                # Subtract 1 here to map 1-based ID to 0-based PyTorch index
                true_spk_emb += self.spk_embed(mix_id.squeeze(-1) - 1) * v
        elif spk_id is not None:
            # Subtract 1 here as well
            true_spk_emb = self.spk_embed(spk_id.squeeze(-1) - 1)
        else:
            true_spk_emb = torch.zeros((units.shape[0], 256), device=units.device)

        # 2. Setup CFG embeddings
        # We only zero out the embedding for the Reflow model during training.
        if torch.is_tensor(drop_spk):
            drop_mask = drop_spk.to(device=true_spk_emb.device, dtype=torch.bool).reshape(-1, 1)
            reflow_spk_emb = torch.where(drop_mask, torch.zeros_like(true_spk_emb), true_spk_emb)
        else:
            reflow_spk_emb = torch.zeros_like(true_spk_emb) if drop_spk else true_spk_emb
        # For inference CFG, we need a null condition
        null_spk_emb = torch.zeros_like(true_spk_emb) if infer else None

        units_t = units.transpose(1, 2)                   # [B, n_unit, T]
        cleaned_units = self.shared_unit_encoder(units_t) # [B, n_unit, T]
        cleaned_units_for_ddsp = cleaned_units.transpose(1, 2)

        # 3. Pass the TRUE spk_id to DDSP so it generates the correct voice base
        ddsp_wav, hidden = self.ddsp_model(cleaned_units_for_ddsp, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, infer=infer)

        start_frame = int(silence_front * self.sampling_rate / self.block_size)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav[:, start_frame * self.block_size:])
        else:
            ddsp_mel = None

        # CFG drops speaker identity only; content, pitch, and timing remain conditioned.
        reflow_cond_mel = ddsp_mel

        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)

            # 5. Use TRUE speaker embedding for F0 Predictor
            if self.f0_predictor is not None:
                spk_emb_expanded = true_spk_emb.unsqueeze(-1).expand(-1, -1, cleaned_units.size(2))
                f0_pred_input = torch.cat([cleaned_units, spk_emb_expanded], dim=1)
                pred_log_f0 = self.f0_predictor(f0_pred_input).transpose(1, 2)
                gt_log_f0 = torch.log(f0.clamp_min(1e-5))
                voiced = f0 > 0
                f0_loss = F.l1_loss(pred_log_f0[voiced], gt_log_f0[voiced]) if voiced.any() else pred_log_f0.sum() * 0.0
            else:
                f0_loss = torch.zeros((), device=units.device, dtype=ddsp_loss.dtype)

            # 6. Use the CFG-dropped variables for Reflow
            if t_start < 1.0:
                reflow_loss = self.reflow_model(
                    condition=reflow_cond_mel,
                    gt_spec=gt_spec,
                    global_cond=reflow_spk_emb,
                    t_start=t_start,
                    infer=False,
                    teacher_velocity_fn=teacher_velocity_fn,
                    return_self_flow_loss=return_self_flow_loss
                )
            else:
                reflow_loss = torch.tensor(0.0, device=units.device)
                if return_self_flow_loss:
                    reflow_loss = (reflow_loss, reflow_loss.clone())
            if return_self_flow_loss:
                reflow_loss, self_flow_loss = reflow_loss
                return ddsp_loss, reflow_loss, f0_loss, self_flow_loss
            return ddsp_loss, reflow_loss, f0_loss
        else:
            if gt_spec is not None and ddsp_mel is None: ddsp_mel = gt_spec
            if t_start < 1.0:
                mel = self.reflow_model(ddsp_mel, gt_spec=gt_spec, global_cond=true_spk_emb, infer=True, infer_step=infer_step, method=method, t_start=t_start, use_tqdm=use_tqdm, cfg_scale=cfg_scale, null_global_cond=null_spk_emb)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0[:, -mel.shape[1]:])
            else:
                return mel

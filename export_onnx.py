import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm
from ddsp.model_conformer_naive import ConformerNaiveEncoder
from onnxruntime import InferenceSession


class LinearSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
        mode="pow2_sqrt",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode
        self.return_complex = False

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, y):
        if y.ndim == 3:
            y = y.squeeze(1)
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (self.win_length - self.hop_length) // 2,
                (self.win_length - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=self.return_complex,
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=False,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)
        from librosa.filters import mel as librosa_mel_fn

        mel_basis = torch.from_numpy(
            librosa_mel_fn(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
            )
        )
        self.register_buffer("mel_basis", mel_basis, persistent=False)

    def forward(self, x):
        linear = self.spectrogram(x)
        spec = torch.matmul(self.mel_basis, linear)
        return torch.log(torch.clamp(spec, min=1e-5))


class iSTFT(nn.Module):
    def __init__(
        self,
        win_len=1024,
        win_hop=512,
        fft_len=1024,
        window=None,
        enframe_mode="continue",
        win_sqrt=False,
    ):
        """
        Implement of STFT using 1D convolution and 1D transpose convolutions.
        Implement of framing the signal in 2 ways, `break` and `continue`.
        `break` method is a kaldi-like framing.
        `continue` method is a librosa-like framing.
        More information about `perfect reconstruction`:
        1. https://ww2.mathworks.cn/help/signal/ref/stft.html
        2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
        Args:
            win_len (int): Number of points in one frame.  Defaults to 1024.
            win_hop (int): Number of framing stride. Defaults to 512.
            fft_len (int): Number of DFT points. Defaults to 1024.
            enframe_mode (str, optional): `break` and `continue`. Defaults to 'continue'.
            window (tensor, optional): The window tensor. Defaults to hann window.
            win_sqrt (bool, optional): using square root window. Defaults to True.
        """
        super(iSTFT, self).__init__()
        assert enframe_mode in ["break", "continue"]
        assert fft_len >= win_len
        self.win_len = win_len
        self.win_hop = win_hop
        self.fft_len = fft_len
        self.mode = enframe_mode
        self.win_sqrt = win_sqrt
        self.pad_amount = self.fft_len // 2

        if window is None:
            window = torch.hann_window(win_len)
        ifft_k, ola_k = self.__init_kernel__(window)
        self.register_buffer("ifft_k", ifft_k, persistent=False)
        self.register_buffer("ola_k", ola_k, persistent=False)

    def __init_kernel__(self, window):
        """
        Generate enframe_kernel, fft_kernel, ifft_kernel and overlap-add kernel.
        ** enframe_kernel: Using conv1d layer and identity matrix.
        ** fft_kernel: Using linear layer for matrix multiplication. In fact,
        enframe_kernel and fft_kernel can be combined, But for the sake of
        readability, I took the two apart.
        ** ifft_kernel, pinv of fft_kernel.
        ** overlap-add kernel, just like enframe_kernel, but transposed.
        Returns:
            tuple: four kernels.
        """
        tmp = torch.fft.rfft(torch.eye(self.fft_len))
        fft_kernel = torch.stack([tmp.real, tmp.imag], dim=2)
        if self.mode == "break":
            fft_kernel = fft_kernel[: self.win_len]
        fft_kernel = torch.cat((fft_kernel[:, :, 0], fft_kernel[:, :, 1]), dim=1)
        ifft_kernel = torch.pinverse(fft_kernel)[:, None, :]

        if self.mode == "continue":
            left_pad = (self.fft_len - self.win_len) // 2
            right_pad = left_pad + (self.fft_len - self.win_len) % 2
            window = F.pad(window, (left_pad, right_pad))
        if self.win_sqrt:
            self.padded_window = window
            window = torch.sqrt(window)
        else:
            self.padded_window = window**2

        ifft_kernel = ifft_kernel * window
        ola_kernel = torch.eye(self.fft_len)[: self.win_len, None, :]
        if self.mode == "continue":
            ola_kernel = torch.eye(self.fft_len)[:, None, : self.fft_len]
        return ifft_kernel, ola_kernel

    def forward(self, spec):
        """Call the inverse STFT (iSTFT), given tensors produced
        by the `transform` function.
        Args:
            spec (tensors): Input tensor with shape
            complex [num_batch, num_frequencies, num_frames]
            or real [num_batch, num_frequencies, num_frames, 2]
            length (int): Expected number of samples in the output audio.
        Returns:
            tensors: Reconstructed audio given magnitude and phase. Of
                shape [num_batch, num_samples]
        """
        if torch.is_complex(spec):
            real, imag = spec.real, spec.imag
        else:
            assert spec.size(-1) == 2
            real, imag = spec[..., 0], spec[..., 1]

        inputs = torch.cat([real, imag], dim=1)
        outputs = F.conv_transpose1d(inputs, self.ifft_k, stride=self.win_hop)
        t = (self.padded_window[None, :, None]).repeat(1, 1, inputs.size(-1))
        t = t.to(inputs.device)
        coff = F.conv_transpose1d(t, self.ola_k, stride=self.win_hop)
        rm_start, rm_end = self.pad_amount, -self.pad_amount
        outputs = outputs[..., rm_start:rm_end]
        coff = coff[..., rm_start:rm_end]
        coff = torch.where(coff > 1e-8, coff, torch.ones_like(coff))
        outputs /= coff
        return outputs.squeeze(dim=1)


def split_to_(tensor, tensor_splits):
    tensors = torch.split(tensor, tensor_splits, dim=-1)
    return tensors


class Unit2Control(nn.Module):
    def __init__(
        self,
        input_channel,
        block_size,
        n_spk,
        output_splits,
        num_layers=3,
        dim_model=256,
        use_norm=False,
        use_attention=False,
        use_pitch_aug=False,
        use_f0_conditioning=False,
    ):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(2, dim_model) if use_f0_conditioning else None
        self.volume_embed = nn.Linear(1, dim_model)
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, dim_model)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, dim_model, bias=False)
        else:
            self.aug_shift_embed = None

        self.stack = nn.Sequential(
            weight_norm(nn.Conv1d(input_channel, 512, 3, 1, 1)),
            nn.PReLU(num_parameters=512),
            weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)),
        )
        self.stack2 = nn.Sequential(
            weight_norm(nn.Conv1d(2 * block_size, 512, 3, 1, 1)),
            nn.PReLU(num_parameters=512),
            weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)),
        )
        self.decoder = ConformerNaiveEncoder(
            num_layers=num_layers,
            num_heads=8,
            dim_model=dim_model,
            use_norm=use_norm,
            conv_only=not use_attention,
            conv_dropout=0,
            atten_dropout=0.1,
        )
        self.norm = nn.LayerNorm(dim_model)
        self.n_out = sum([v for k, v in output_splits.items()])
        self.o_sp_k = [k for k, v in output_splits.items()]
        self.o_sp = [v for k, v in output_splits.items()]
        self.dense_out = weight_norm(nn.Linear(dim_model, self.n_out))
        self.gin_channels = dim_model

    def export_chara_mix(self, n_spk):
        speaker_map = torch.zeros((n_spk, 1, 1, self.gin_channels))
        for i in range(n_spk):
            speaker_map[i] = self.spk_embed(torch.LongTensor([[i]]))
        speaker_map = speaker_map.unsqueeze(0)
        self.register_buffer("speaker_map", speaker_map)

    def forward(self, units, source, noise, volume, f0, voiced, g, aug_shift):
        exciter = torch.cat((source, noise), dim=-1).transpose(1, 2)
        x = self.stack(units.transpose(1, 2)) + self.stack2(exciter)
        x = x.transpose(1, 2) + self.volume_embed(volume)

        if self.f0_embed is not None:
            log_f0 = torch.log2(f0.clamp_min(1.0)) / 10.0
            x = x + self.f0_embed(torch.cat((log_f0, voiced), dim=-1))

        if self.n_spk is not None and self.n_spk > 1:
            g = g.permute(2, 0, 1)  # [S, B, N]
            g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2], 1))  # [1, S, B, N, 1]
            g = g * self.speaker_map  # [1, S, B, N, H]
            g = torch.sum(g, dim=1).squeeze(0)  # [B, N, H]
            x = x + g

        if self.aug_shift_embed is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)

        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        return split_to_(e, self.o_sp)


class CombSubSuperFast(torch.nn.Module):
    def __init__(
        self,
        sampling_rate,
        block_size,
        win_length,
        n_unit=256,
        n_spk=1,
        num_layers=3,
        dim_model=256,
        use_norm=False,
        use_attention=False,
        use_pitch_aug=False,
        use_f0_conditioning=False,
        out_dims=128,
        spec_min=-12,
        spec_max=2,
        f_min=40,
        f_max=16000,
    ):
        super().__init__()

        print(" [DDSP Model] Combtooth Subtractive Synthesiser")
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        self.sr = sampling_rate
        self.bs = block_size
        self.wl = win_length
        # Unit2Control
        split_map = {
            "harmonic_magnitude": win_length // 2 + 1,
            "harmonic_phase": win_length // 2 + 1,
            "noise_magnitude": win_length // 2 + 1,
            "noise_phase": win_length // 2 + 1,
        }
        self.unit2ctrl = Unit2Control(
            n_unit,
            block_size,
            n_spk,
            split_map,
            num_layers=num_layers,
            dim_model=dim_model,
            use_norm=use_norm,
            use_attention=use_attention,
            use_pitch_aug=use_pitch_aug,
            use_f0_conditioning=use_f0_conditioning,
        )

        self.istft_method = iSTFT(
            win_len=win_length,
            win_hop=block_size,
            fft_len=win_length,
            window=self.window,
        )

        self.melext = LogMelSpectrogram(
            sampling_rate,
            win_length,
            win_length,
            block_size,
            out_dims,
            False,
            f_min,
            f_max,
        )

        self.spec_max = spec_max
        self.spec_min = spec_min

    def fast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = self.msinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)
        return combtooth

    @staticmethod
    def msinc(input):
        input = np.pi * input
        return torch.where(
            torch.abs(input) < 1e-5, torch.ones_like(input), torch.sin(input) / input
        )

    @staticmethod
    def complex_exp(real, imag):
        mod = torch.exp(real)
        rp = torch.cos(imag)
        ip = torch.sin(imag)
        return torch.cat((rp, ip), dim=-1) * mod

    @staticmethod
    def complex_mul(left, right):
        assert len(left.shape) == len(right.shape)
        real = (
            left[:, :, :, 0] * right[:, :, :, 0] - left[:, :, :, 1] * right[:, :, :, 1]
        ).unsqueeze(-1)
        imag = (
            left[:, :, :, 0] * right[:, :, :, 1] + left[:, :, :, 1] * right[:, :, :, 0]
        ).unsqueeze(-1)
        return torch.cat((real, imag), dim=-1)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def sfast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = torch.sinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)
        return combtooth

    def sforward(
        self,
        units_frames,
        mel2ph,
        f0_frames,
        voiced_frames,
        volume_frames,
        aug_shift,
        g=None,
        noise=None,
    ):
        """
        units_frames: B x n_frames x n_unit
        f0_frames: B x n_frames x 1
        volume_frames: B x n_frames x 1
        spk_id: B x 1
        """

        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, units_frames.shape[-1]])
        units_frames = torch.gather(units_frames, 1, mel2ph_)

        volume_frames = volume_frames.unsqueeze(-1)

        combtooth = self.fast_source_gen(f0_frames.unsqueeze(-1))
        combtooth_frames = combtooth.unfold(1, self.block_size, self.block_size)

        noise_frames = noise.unfold(1, self.block_size, self.block_size)

        # parameter prediction
        harmonic_magnitude, harmonic_phase, noise_magnitude, noise_phase = (
            self.unit2ctrl(
                units_frames,
                combtooth_frames,
                noise_frames,
                volume_frames,
                f0_frames.unsqueeze(-1),
                voiced_frames.unsqueeze(-1),
                g,
                aug_shift,
            )
        )
        harmonic_magnitude = harmonic_magnitude.clamp(-12.0, 6.0)
        noise_magnitude = noise_magnitude.clamp(-12.0, 6.0)
        src_filter = torch.exp(harmonic_magnitude + 1.0j * np.pi * harmonic_phase)
        src_filter = torch.cat((src_filter, src_filter[:, -1:, :]), 1)
        noise_filter = torch.exp(noise_magnitude + 1.0j * np.pi * noise_phase) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:, -1:, :]), 1)

        # harmonic part filter
        if combtooth.shape[-1] > self.win_length // 2:
            pad_mode = "reflect"
        else:
            pad_mode = "constant"
        combtooth_stft = torch.stft(
            combtooth,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.block_size,
            window=self.window,
            center=True,
            return_complex=True,
            pad_mode=pad_mode,
        )

        # noise part filter
        noise_stft = torch.stft(
            noise,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.block_size,
            window=self.window,
            center=True,
            return_complex=True,
            pad_mode=pad_mode,
        )

        # apply the filters
        signal_stft = combtooth_stft * src_filter.permute(
            0, 2, 1
        ) + noise_stft * noise_filter.permute(0, 2, 1)

        # take the istft to resynthesize audio.
        signal = torch.istft(
            signal_stft,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.block_size,
            window=self.window,
            center=True,
        )

        return self.melext(signal)

    def tforward(
        self,
        units_frames,
        mel2ph,
        f0_frames,
        voiced_frames,
        volume_frames,
        aug_shift,
        g=None,
        noise=None,
    ):
        """
        units_frames: B x n_frames x n_unit
        f0_frames: B x n_frames
        volume_frames: B x n_frames
        """
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, units_frames.shape[-1]])
        units_frames = torch.gather(units_frames, 1, mel2ph_)

        volume_frames = volume_frames.unsqueeze(-1)

        combtooth = self.fast_source_gen(f0_frames.unsqueeze(-1))
        combtooth_frames = combtooth.reshape(combtooth.size(0), -1, self.block_size)

        noise_frames = noise.reshape(noise.size(0), -1, self.block_size)

        harmonic_magnitude, harmonic_phase, noise_magnitude, noise_phase = (
            self.unit2ctrl(
                units_frames,
                combtooth_frames,
                noise_frames,
                volume_frames,
                f0_frames.unsqueeze(-1),
                voiced_frames.unsqueeze(-1),
                g,
                aug_shift,
            )
        )
        harmonic_magnitude = harmonic_magnitude.clamp(-12.0, 6.0)
        noise_magnitude = noise_magnitude.clamp(-12.0, 6.0)

        src_filter = self.complex_exp(
            harmonic_magnitude.unsqueeze(-1),
            torch.ones_like(harmonic_magnitude, dtype=torch.float32).unsqueeze(-1)
            * np.pi
            * harmonic_phase.unsqueeze(-1),
        )
        src_filter = torch.cat((src_filter, src_filter[:, -1:, :]), 1)

        noise_filter = (
            self.complex_exp(
                noise_magnitude.unsqueeze(-1),
                torch.ones_like(noise_magnitude, dtype=torch.float32).unsqueeze(-1)
                * np.pi
                * noise_phase.unsqueeze(-1),
            )
            / 128
        )
        noise_filter = torch.cat((noise_filter, noise_filter[:, -1:, :]), 1)

        # harmonic part filter
        if combtooth.shape[-1] > self.win_length // 2:
            pad_mode = "reflect"
        else:
            pad_mode = "constant"
        combtooth_stft = torch.stft(
            combtooth,
            n_fft=self.wl,
            win_length=self.wl,
            hop_length=self.bs,
            window=self.window,
            center=True,
            return_complex=False,
            pad_mode=pad_mode,
        )

        # noise part filter
        noise_stft = torch.stft(
            noise,
            n_fft=self.wl,
            win_length=self.wl,
            hop_length=self.bs,
            window=self.window,
            center=True,
            return_complex=False,
            pad_mode=pad_mode,
        )

        # apply the filters
        signal_stft = self.complex_mul(
            combtooth_stft, src_filter.permute(0, 2, 1, 3)
        ) + self.complex_mul(noise_stft, noise_filter.permute(0, 2, 1, 3))

        signal = self.istft_method(signal_stft)

        return self.melext(signal)

    def forward(
        self,
        units_frames,
        mel2ph,
        f0_frames,
        voiced_frames,
        volume_frames,
        aug_shift,
        g=None,
        noise=None,
    ):
        """
        units_frames: B x n_frames x n_unit
        f0_frames: B x n_frames
        volume_frames: B x n_frames
        """
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, units_frames.shape[-1]])
        units_frames = torch.gather(units_frames, 1, mel2ph_)

        volume_frames = volume_frames.unsqueeze(-1)

        combtooth = self.fast_source_gen(f0_frames.unsqueeze(-1))
        combtooth_frames = combtooth.reshape(combtooth.size(0), -1, self.block_size)

        noise_frames = noise.reshape(noise.size(0), -1, self.block_size)

        harmonic_magnitude, harmonic_phase, noise_magnitude, noise_phase = (
            self.unit2ctrl(
                units_frames,
                combtooth_frames,
                noise_frames,
                volume_frames,
                f0_frames.unsqueeze(-1),
                voiced_frames.unsqueeze(-1),
                g,
                aug_shift,
            )
        )
        harmonic_magnitude = harmonic_magnitude.clamp(-12.0, 6.0)
        noise_magnitude = noise_magnitude.clamp(-12.0, 6.0)

        src_filter = self.complex_exp(
            harmonic_magnitude.unsqueeze(-1),
            torch.ones_like(harmonic_magnitude, dtype=torch.float32).unsqueeze(-1)
            * np.pi
            * harmonic_phase.unsqueeze(-1),
        )
        src_filter = torch.cat((src_filter, src_filter[:, -1:, :]), 1)

        noise_filter = (
            self.complex_exp(
                noise_magnitude.unsqueeze(-1),
                torch.ones_like(noise_magnitude, dtype=torch.float32).unsqueeze(-1)
                * np.pi
                * noise_phase.unsqueeze(-1),
            )
            / 128
        )
        noise_filter = torch.cat((noise_filter, noise_filter[:, -1:, :]), 1)

        # harmonic part filter
        if combtooth.shape[-1] > self.win_length // 2:
            pad_mode = "reflect"
        else:
            pad_mode = "constant"
        combtooth_stft = torch.stft(
            combtooth,
            n_fft=self.wl,
            win_length=self.wl,
            hop_length=self.bs,
            window=self.window,
            center=True,
            return_complex=False,
            pad_mode=pad_mode,
        )

        # noise part filter
        noise_stft = torch.stft(
            noise,
            n_fft=self.wl,
            win_length=self.wl,
            hop_length=self.bs,
            window=self.window,
            center=True,
            return_complex=False,
            pad_mode=pad_mode,
        )

        # apply the filters
        signal_stft = self.complex_mul(
            combtooth_stft, src_filter.permute(0, 2, 1, 3)
        ) + self.complex_mul(noise_stft, noise_filter.permute(0, 2, 1, 3))

        signal = self.istft_method(signal_stft)
        mel = self.melext(signal)

        return self.norm_spec(mel).unsqueeze(1), mel


class OnnxConditionerBase(nn.Module):
    def __init__(self, runtime_model, ddsp_model):
        super().__init__()
        self.shared_unit_encoder = runtime_model.shared_unit_encoder
        self.spk_embed = runtime_model.spk_embed
        self.ddsp_model = ddsp_model
        self.block_size = runtime_model.block_size

    def condition(self, units, mel2ph, f0, voiced, volume, aug_shift, noise, spk_mix):
        units = self.shared_unit_encoder(units.transpose(1, 2)).transpose(1, 2)
        x, cond = self.ddsp_model(
            units, mel2ph, f0, voiced, volume, aug_shift, g=spk_mix, noise=noise
        )
        if spk_mix is None:
            global_cond = (
                self.spk_embed.weight[0].unsqueeze(0).expand(units.size(0), -1)
            )
        else:
            global_cond = torch.matmul(spk_mix, self.spk_embed.weight)
        return x, cond, global_cond


class OnnxConditionerSingle(OnnxConditionerBase):
    def forward(self, units, mel2ph, f0, voiced, volume, aug_shift, noise):
        return self.condition(units, mel2ph, f0, voiced, volume, aug_shift, noise, None)


class OnnxConditionerMulti(OnnxConditionerBase):
    def forward(self, units, mel2ph, f0, voiced, volume, aug_shift, spk_mix, noise):
        return self.condition(
            units, mel2ph, f0, voiced, volume, aug_shift, noise, spk_mix
        )


def export_onnx(model_path, output_path):
    from reflow.vocoder import load_model_vocoder

    os.makedirs(output_path, exist_ok=True)
    runtime_model, vocoder, args = load_model_vocoder(model_path, device="cpu")

    out_dims = vocoder.dimension
    spec_min = args.model.get("spec_min", -12)
    spec_max = args.model.get("spec_max", 2)
    vocoder_config = vocoder.vocoder.h
    ddsp_model = CombSubSuperFast(
        sampling_rate=args.data.sampling_rate,
        block_size=args.data.block_size,
        win_length=args.model.win_length,
        n_unit=args.data.encoder_out_channels,
        n_spk=args.model.n_spk,
        num_layers=args.model.n_aux_layers,
        dim_model=args.model.n_aux_chans,
        use_norm=args.model.use_norm,
        use_attention=args.model.use_attention,
        use_pitch_aug=args.model.use_pitch_aug,
        use_f0_conditioning=args.model.get("use_f0_conditioning", False),
        out_dims=out_dims,
        spec_min=spec_min,
        spec_max=spec_max,
        f_min=vocoder_config.get("fmin", 40),
        f_max=vocoder_config.get("fmax", args.data.sampling_rate // 2),
    )
    ddsp_model.unit2ctrl.load_state_dict(
        runtime_model.ddsp_model.unit2ctrl.state_dict()
    )

    n_spk = args.model.n_spk
    if n_spk is not None and n_spk > 1:
        ddsp_model.unit2ctrl.export_chara_mix(n_spk)
        conditioner = OnnxConditionerMulti(runtime_model, ddsp_model)
    else:
        conditioner = OnnxConditionerSingle(runtime_model, ddsp_model)
    conditioner.eval()

    frame_c = 25
    hu = torch.randn((1, frame_c, args.data.encoder_out_channels))
    mel2ph = torch.arange(0, frame_c).long().unsqueeze(0)
    f0 = torch.rand(1, frame_c) * 300 + 100
    voiced = torch.ones(1, frame_c)
    vol = torch.rand(1, frame_c)
    aug_shift = torch.zeros(1, 1, 1)
    randn_input = torch.randn(1, frame_c * runtime_model.block_size)

    if n_spk is not None and n_spk > 1:
        spk_mix = torch.full((1, frame_c, n_spk), 1.0 / n_spk)
        conditioner_inputs = (
            hu,
            mel2ph,
            f0,
            voiced,
            vol,
            aug_shift,
            spk_mix,
            randn_input,
        )
        conditioner_input_names = [
            "hubert",
            "mel2ph",
            "f0",
            "voiced",
            "volume",
            "aug_shift",
            "spk_mix",
            "randn",
        ]
    else:
        conditioner_inputs = (hu, mel2ph, f0, voiced, vol, aug_shift, randn_input)
        conditioner_input_names = [
            "hubert",
            "mel2ph",
            "f0",
            "voiced",
            "volume",
            "aug_shift",
            "randn",
        ]

    dynamic_axes = {
        "hubert": {0: "batch", 1: "unit_frame"},
        "mel2ph": {0: "batch", 1: "frame"},
        "f0": {0: "batch", 1: "frame"},
        "voiced": {0: "batch", 1: "frame"},
        "volume": {0: "batch", 1: "frame"},
        "aug_shift": {0: "batch"},
        "randn": {0: "batch", 1: "audio_length"},
        "x": {0: "batch", 3: "frame"},
        "cond": {0: "batch", 2: "frame"},
        "global_cond": {0: "batch"},
    }
    if n_spk is not None and n_spk > 1:
        dynamic_axes["spk_mix"] = {0: "batch", 1: "frame"}
        dynamic_axes["global_cond"][1] = "frame"

    with torch.no_grad():
        outtest = conditioner(*conditioner_inputs)
    encoder_path = os.path.join(output_path, "encoder.onnx")
    torch.onnx.export(
        conditioner,
        conditioner_inputs,
        encoder_path,
        dynamic_axes=dynamic_axes,
        do_constant_folding=False,
        dynamo=False,
        opset_version=18,
        verbose=False,
        input_names=conditioner_input_names,
        output_names=["x", "cond", "global_cond"],
    )

    t = torch.zeros((1,), dtype=torch.float32)
    velocity_path = os.path.join(output_path, "velocity.onnx")
    velocity_dynamic_axes = {
        "x": {0: "batch", 3: "frame"},
        "t": {0: "batch"},
        "cond": {0: "batch", 2: "frame"},
        "global_cond": {0: "batch"},
        "o": {0: "batch", 3: "frame"},
    }
    if outtest[2].dim() == 3:
        velocity_dynamic_axes["global_cond"][1] = "frame"
    torch.onnx.export(
        runtime_model.reflow_model.velocity_fn,
        (outtest[0], t, outtest[1], outtest[2]),
        velocity_path,
        input_names=["x", "t", "cond", "global_cond"],
        output_names=["o"],
        dynamic_axes=velocity_dynamic_axes,
        dynamo=False,
        opset_version=18,
    )

    encoder_session = InferenceSession(encoder_path)
    ort_inputs = {
        name: value.detach().cpu().numpy()
        for name, value in zip(conditioner_input_names, conditioner_inputs)
    }
    ort_outputs = encoder_session.run(None, ort_inputs)
    for name, expected, actual in zip(
        ("x", "cond", "global_cond"), outtest, ort_outputs
    ):
        np.testing.assert_allclose(
            actual,
            expected.detach().cpu().numpy(),
            rtol=2e-3,
            atol=2e-3,
            err_msg=f"ONNX conditioner output mismatch: {name}",
        )

    return args.data.encoder_out_channels, runtime_model.block_size, n_spk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a DDSP-Reflow checkpoint to ONNX"
    )
    parser.add_argument("-m", "--model", required=True, help="checkpoint path")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    cli_args = parser.parse_args()
    export_onnx(cli_args.model, cli_args.output)

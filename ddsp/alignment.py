import torch
import torch.nn.functional as F


def pad_short_audio(audio, min_samples=400):
    """Right-pad encoder-rate audio to the encoder's minimum input length."""
    if audio.size(-1) >= min_samples:
        return audio
    return F.pad(audio, (0, min_samples - audio.size(-1)))


def align_units(units, n_frames, ratio):
    """Nearest-neighbor align unit features while preserving every batch."""
    index = torch.round(
        ratio * torch.arange(n_frames, device=units.device)
    ).long()
    index = torch.clamp(index, max=units.size(1) - 1)
    index = index[None, :, None].expand(
        units.size(0), index.size(0), units.size(-1))
    return torch.gather(units, 1, index)

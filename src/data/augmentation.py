import torch

def audio_augmentation(audio: torch.Tensor) -> torch.Tensor:
    audio = audio.clone()

    if torch.rand(()).item() < 0.5:
        gain = torch.empty(
            (), device=audio.device
        ).uniform_(0.8, 1.2)

        audio = audio * gain

    if torch.rand(()).item() < 0.3:
        signal_rms = audio.square().mean().sqrt()

        snr_db = torch.empty(
            (), device=audio.device
        ).uniform_(20.0, 35.0)

        noise = torch.randn_like(audio)

        noise_rms = noise.square().mean().sqrt()
        noise = noise / noise_rms.clamp_min(1e-8)

        target_noise_rms = signal_rms * torch.pow(
            10.0, -snr_db / 20.0
        )

        audio = audio + noise * target_noise_rms

    return audio

def log_mel_augmentation(mel_audio: torch.Tensor) -> torch.Tensor:
    mel_audio = mel_audio.clone()

    n_mels, time_steps = mel_audio.shape

    mask_value = mel_audio.mean()

    if torch.rand(()) < 0.5:
        max_width = min(9, n_mels)

        if max_width > 0:
            width = torch.randint(1, max_width + 1, ()).item()
            start = torch.randint(0, n_mels - width + 1, ()).item()

            mel_audio[start:start + width, :] = mask_value

    if torch.rand(()) < 0.5:
        max_width = min(30, int(time_steps * 0.05))

        if max_width > 0:
            width = torch.randint(1, max_width + 1, ()).item()
            start = torch.randint(0, time_steps - width + 1, ()).item()

            mel_audio[:, start:start + width] = mask_value

    return mel_audio
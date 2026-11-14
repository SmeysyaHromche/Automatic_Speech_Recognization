
import torch
import torch.nn.functional as F

def collate_fn(batch):
    audios, texts = zip(*batch)

    lengths = torch.tensor([audio.shape[-1] for audio in audios], dtype=torch.long)

    max_length = lengths.max().item()

    padded_audios = []

    for audio in audios:
        padding = max_length - audio.shape[-1]

        padded_audio = F.pad(
            audio,
            (0, padding),
            value=0.0
        )

        padded_audios.append(padded_audio)

    audios = torch.stack(padded_audios)

    return audios, texts, lengths

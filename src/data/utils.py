from .audio_dataset import AudioDataset
from src.config import DataConfig, TrainConfig
from torch.utils.data import DataLoader
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

def get_dataloader(
    config:DataConfig,
    batch_size:int,
    shuffle:bool=False,
    train:bool=False
) -> DataLoader:
    dataset = AudioDataset(
        config=config, 
        shuffle=shuffle,
        train=train
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )

    return dataloader

def get_train_valid_dataloaders(config:TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_dataloader = get_dataloader(
        config.train_data, 
        config.batch_size,
        shuffle=True,
        train=True
    )

    valid_dataloader = get_dataloader(
        config.validation_data,
        config.batch_size
    )

    return train_dataloader, valid_dataloader
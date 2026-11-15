
from .augmentation import audio_augmentation, mel_augmentation
from src.config import DataConfig
from torch.utils.data import IterableDataset
import datasets
import torch
import torchaudio

class AudioDataset(IterableDataset):
    
    DATASET_PATH = "openslr/librispeech_asr"

    def __init__(
        self, 
        config:DataConfig, 
        shuffle:bool=False, 
        train:bool=False
    ):
        super().__init__()
        
        self.config = config
        self.train = train
        
        self.dataset = datasets.load_dataset(
            self.DATASET_PATH, 
            self.config.dataset_name, 
            split=self.config.split, 
            streaming=True)

        if shuffle:
            self.dataset = self.dataset.shuffle(seed=self.config.shuffle_seeds, buffer_size=100)
        
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.mel.sr,
            n_fft=self.config.mel.n_fft,
            hop_length=self.config.mel.hop_len,
            n_mels=self.config.mel.n_mels
        )
    
    def __iter__(self):
        for next_sample in self.dataset:
            audio, text = next_sample["audio"], next_sample["text"]
            audio_sr = audio["sampling_rate"]
            audio = torch.as_tensor(audio["array"], dtype=torch.float32)
            
            if audio_sr != self.config.mel.sr:
                audio = torchaudio.functional.resample(
                    audio,
                    orig_freq=audio_sr,
                    new_freq=self.config.mel.sr
                )
            if self.train and self.config.augmentation.enable:
                audio = audio_augmentation(audio)
            
            mel_audio = self.mel(audio)
            mel_audio = torch.log(mel_audio.clamp_min(1e-6))
            if self.train and self.config.augmentation.mel_enable:
                mel_audio = mel_augmentation(mel_audio)
            mel_audio = mel_audio.unsqueeze(0)
            
            yield mel_audio, text
        
         
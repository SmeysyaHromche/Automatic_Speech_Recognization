
from .augmentation import audio_augmentation, mel_augmentation
from src.config import DataConfig
from torch.utils.data import IterableDataset
import datasets
import torch
from torchaudio.transforms import MelSpectrogram

class AudioDataset(IterableDataset):
    
    DATASET_PATH = "openslr/librispeech_asr"

    def __init__(self, config:DataConfig):
        super().__init__()
        
        self.config = config
        
        self.dataset = datasets.load_dataset(
            self.DATASET_PATH, 
            self.config.dataset_name, 
            split=self.config.split, 
            streaming=True)

        self.iter = None
        
        self.shuffle()

        self.reset_iter()

        self.mel = MelSpectrogram(
            sample_rate=self.config.mel.sr,
            n_fft=self.config.mel.n_fft,
            hop_length=self.config.mel.hop_len,
            n_mels=self.config.mel.n_mels
        )

    def shuffle(self) -> None:
        self.dataset = self.dataset.shuffle(self.config.shuffle_seeds)
    
    def reset_iter(self) -> None:
        self.iter = iter(self.dataset)
    
    def __iter__(self):
        next_sample = next(self.iter)

        audio, text = next_sample["audio"], next_sample["text"]
        audio = torch.as_tensor(audio["array"], dtype=torch.float32)
        
        if self.config.augmentation.enable:
            audio = audio_augmentation(audio)
        
        mel_audio = self.mel(audio) 
        
        if self.config.augmentation.mel_enable:
            mel_audio = mel_augmentation(mel_audio)
            mel_audio = torch.math.log(mel_audio + 1e-6)

        return mel_audio, text
        
         
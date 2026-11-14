from .augmentation_config import AugmentationConfig
from .mel_config import MelConfig
from pydantic import BaseModel

class DataConfig(BaseModel):
    dataset_name: str
    split: str
    shuffle_seeds: int
    
    augmentation: AugmentationConfig
    
    mel: MelConfig
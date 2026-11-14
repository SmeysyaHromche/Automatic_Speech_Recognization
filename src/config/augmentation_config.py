from pydantic import BaseModel

class AugmentationConfig(BaseModel):
    enable:bool
    mel_enable:bool
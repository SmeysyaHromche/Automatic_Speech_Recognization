from pydantic import BaseModel

class MelConfig(BaseModel):
    enable:bool
    sr:int
    n_fft:int
    hop_len:int
    n_mels:int
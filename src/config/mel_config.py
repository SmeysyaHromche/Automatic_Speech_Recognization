from pydantic import BaseModel

class MelConfig(BaseModel):
    sr:int
    n_fft:int
    hop_len:int
    n_mels:int
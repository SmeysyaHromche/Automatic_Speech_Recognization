from pydantic import BaseModel

class EncoderConfig(BaseModel):
    normalization:str
    activation_fn:str

    n_mel:int
    d_model:int
    channels:tuple[int, ...]
    strides:tuple[int, ...]
    res_blocks_per_stage:tuple[int, ...]
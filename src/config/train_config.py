from .data_config import DataConfig
from pydantic import BaseModel

class TrainConfig(BaseModel):
    batch_size:int
    device:str
    epochs:int
    lr:float

    data: DataConfig

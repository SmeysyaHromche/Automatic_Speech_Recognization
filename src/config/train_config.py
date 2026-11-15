from .data_config import DataConfig
from .logging_config import LoggingConfig
from pydantic import BaseModel

class TrainConfig(BaseModel):
    batch_size:int
    device:str
    epochs:int
    lr:float

    data: DataConfig
    logging_config: LoggingConfig

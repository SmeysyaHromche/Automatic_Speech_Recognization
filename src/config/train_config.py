from .data_config import DataConfig
from .logging_config import LoggingConfig
from pydantic import BaseModel

class TrainConfig(BaseModel):
    batch_size:int
    device:str
    epochs:int
    lr:float
    max_steps_per_epoch:int
    path_to_best_model:str
    path_to_last_model:str

    train_data: DataConfig
    validation_data: DataConfig
    logging_config: LoggingConfig

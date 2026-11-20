from .data_config import DataConfig
from .decoder_config import DecoderConfig
from .encoder_config import EncoderConfig
from .logging_config import LoggingConfig
from pydantic import BaseModel

class TrainConfig(BaseModel):
    batch_size:int
    device:str
    epochs:int
    lr:float
    lr_scheduler:bool
    optimizer:str
    max_steps_per_epoch:int

    encoder:EncoderConfig

    decoder:DecoderConfig
    
    train_data: DataConfig
    
    validation_data: DataConfig
    
    logging_config: LoggingConfig

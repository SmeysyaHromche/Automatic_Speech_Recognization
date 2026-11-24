from .data_config import DataConfig
from .decoder_config import DecoderConfig
from .encoder_config import EncoderConfig
from .logging_config import LoggingConfig
from .search_config import SearchConfig
from pydantic import BaseModel

class TestConfig(BaseModel):
    batch_size:int
    device:str

    path_to_weights:str
    encoder:EncoderConfig
    decoder:DecoderConfig
    search:SearchConfig

    test_data: DataConfig

    logging_config: LoggingConfig

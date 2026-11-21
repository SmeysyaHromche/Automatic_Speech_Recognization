from pydantic import BaseModel

class DecoderConfig(BaseModel):
    rnn:str
    hidden_size:int
    num_layer:int
    norm:bool
    projection:str
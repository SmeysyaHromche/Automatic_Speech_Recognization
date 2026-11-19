from pydantic import BaseModel

class DecoderConfig(BaseModel):
    hidden_size:int
    num_layer:int
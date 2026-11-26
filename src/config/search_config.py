from pydantic import BaseModel

class SearchConfig(BaseModel):
    search_policy:str
    beam_width:int
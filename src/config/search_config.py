from pydantic import BaseModel

class SearchConfig:
    search_policy:str
    beam_width:int
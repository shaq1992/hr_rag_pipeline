from pydantic import BaseModel
from typing import List

class EmbedRequest(BaseModel):
    text: str

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

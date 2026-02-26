from pydantic import BaseModel
from typing import List

class EmbedResponse(BaseModel):
    dense_vector: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]

class RerankResponse(BaseModel):
    scores: List[float]

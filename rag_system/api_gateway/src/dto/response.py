from pydantic import BaseModel
from typing import List, Literal, Optional

class CitationDTO(BaseModel):
    source_document: str
    section_header: str
    page_number: str

class RoutingDecision(BaseModel):
    query_type: Literal['factual', 'procedural', 'comparative', 'out-of-scope']
    reasoning: str

class RetrievedChunk(BaseModel):
    id: str
    content: str
    source_document: str
    section_header: str
    page_number: str
    score: Optional[float] = None

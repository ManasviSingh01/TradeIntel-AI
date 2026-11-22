from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=8)

class Source(BaseModel):
    document: str
    score: float
    excerpt: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    mode: str

class DocumentResponse(BaseModel):
    filename: str
    chunks_added: int

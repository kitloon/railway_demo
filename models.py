from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The AI generated answer")
    topic: str = Field(..., description="Classified topic")
    sources: List[str] = Field(..., description="List of source documents used")

class UrlIngestRequest(BaseModel):
    url: str = Field(..., description="URL to scrape and ingest")

class DeleteSourceRequest(BaseModel):
    source_path: str = Field(..., description="The exact source path to delete")
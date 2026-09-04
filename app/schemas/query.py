from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class SourceResponse(BaseModel):
    source: str
    excerpt: str
    score: float | None
    page: int | None = None
    chunk: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]

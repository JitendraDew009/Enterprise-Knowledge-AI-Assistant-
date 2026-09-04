from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    conversation_id: str | None = None


class ChatSource(BaseModel):
    source: str
    excerpt: str
    score: float | None
    page: int | None = None
    chunk: int | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[ChatSource]

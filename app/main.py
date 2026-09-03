from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import get_settings
from .rag import KnowledgeBase, read_supported_document

app = FastAPI(title="Enterprise Knowledge AI Assistant", version="0.1.0")


@lru_cache
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(get_settings())


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class SourceResponse(BaseModel):
    source: str
    excerpt: str
    score: float | None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: Annotated[UploadFile, File()]) -> dict[str, int | str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")
    try:
        content = read_supported_document(file.filename, await file.read())
        chunks = get_knowledge_base().add_document(file.filename, content)
    except (UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"filename": file.filename, "chunks_indexed": chunks}


@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    answer, sources = get_knowledge_base().answer(request.question)
    return QueryResponse(answer=answer, sources=[SourceResponse(**source.__dict__) for source in sources])

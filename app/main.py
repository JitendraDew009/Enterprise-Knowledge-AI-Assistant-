import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

from .config import get_settings
from .core.logging import RequestLoggingMiddleware, configure_logging
from .core.security import require_user
from .db.session import check_database, get_db
from .ingestion import extract_document_pages, validate_upload
from .rag import DocumentSummary, KnowledgeBase
from .schemas.chat import ChatRequest, ChatResponse
from .services.chat import ConversationNotFoundError, ConversationService

configure_logging()
app = FastAPI(title="Enterprise Knowledge AI Assistant", version="0.1.0")
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in get_settings().cors_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID", "X-User-ID"],
)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, error: Exception) -> JSONResponse:
    logging.getLogger(__name__).exception("unhandled_request_error", extra={"path": request.url.path})
    return JSONResponse(status_code=500, content={"detail": "An internal server error occurred."})


@lru_cache
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(get_settings())


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


class DocumentResponse(BaseModel):
    filename: str
    chunks: int


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def readiness() -> dict[str, str]:
    database_status = "ok" if check_database() else "unavailable"
    overall_status = "ok" if database_status == "ok" else "degraded"
    payload = {"status": overall_status, "database": database_status}
    if overall_status != "ok":
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.post("/documents")
async def upload_document(
    file: Annotated[UploadFile, File()],
    _user_id: Annotated[str, Depends(require_user)],
) -> dict[str, int | str]:
    try:
        raw_content = await file.read(get_settings().max_upload_bytes + 1)
        filename = validate_upload(
            file.filename or "",
            len(raw_content),
            file.content_type,
            get_settings().max_upload_bytes,
        )
        pages = extract_document_pages(filename, raw_content)
        chunks = get_knowledge_base().add_document_pages(filename, pages)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"filename": filename, "chunks_indexed": chunks}


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents(_user_id: Annotated[str, Depends(require_user)]) -> list[DocumentSummary]:
    return get_knowledge_base().list_documents()


@app.delete("/documents/{filename:path}", status_code=204)
def delete_document(filename: str, _user_id: Annotated[str, Depends(require_user)]) -> None:
    if not get_knowledge_base().delete_document(filename):
        raise HTTPException(status_code=404, detail="Document not found.")


@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(
    request: QueryRequest,
    _user_id: Annotated[str, Depends(require_user)],
) -> QueryResponse:
    answer, sources = get_knowledge_base().answer(request.question)
    return QueryResponse(answer=answer, sources=[SourceResponse(**source.__dict__) for source in sources])


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    session: Annotated[Session, Depends(get_db)],
    user_id: Annotated[str, Depends(require_user)],
) -> ChatResponse:
    try:
        return ConversationService(get_settings(), get_knowledge_base(), session).chat(request, user_id)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail="Conversation not found.") from error

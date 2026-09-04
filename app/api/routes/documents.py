import logging
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ...config import get_settings
from ...core.security import require_user
from ...ingestion import extract_document_pages, validate_upload
from ...rag import DocumentSummary
from ...schemas.documents import DocumentResponse
from .. import deps

router = APIRouter(prefix="/documents", tags=["documents"])


def _run_ingestion(task_id: str, filename: str, pages) -> None:
    deps.ingestion_tasks[task_id]["status"] = "processing"
    try:
        chunks = deps.get_knowledge_base().add_document_pages(filename, pages)
        deps.ingestion_tasks[task_id].update(status="completed", chunks_indexed=chunks)
    except Exception:
        logging.getLogger(__name__).exception("background_ingestion_failed")
        deps.ingestion_tasks[task_id].update(status="failed", error="The document could not be indexed.")


@router.post("")
async def upload_document(
    file: Annotated[UploadFile, File()],
    _user_id: Annotated[str, Depends(require_user)],
    background_tasks: BackgroundTasks,
    background: bool = False,
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
        if background:
            task_id = str(uuid4())
            deps.ingestion_tasks[task_id] = {"task_id": task_id, "status": "queued", "filename": filename}
            background_tasks.add_task(_run_ingestion, task_id, filename, pages)
            return JSONResponse(
                status_code=202,
                content={"task_id": task_id, "filename": filename, "status": "queued"},
            )
        chunks = deps.get_knowledge_base().add_document_pages(filename, pages)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"filename": filename, "chunks_indexed": chunks}


@router.get("/tasks/{task_id}")
def ingestion_status(
    task_id: str,
    _user_id: Annotated[str, Depends(require_user)],
) -> dict[str, int | str]:
    task = deps.ingestion_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Ingestion task not found.")
    return task


@router.get("", response_model=list[DocumentResponse])
def list_documents(_user_id: Annotated[str, Depends(require_user)]) -> list[DocumentSummary]:
    return deps.get_knowledge_base().list_documents()


@router.delete("/{filename:path}", status_code=204)
def delete_document(filename: str, _user_id: Annotated[str, Depends(require_user)]) -> None:
    if not deps.get_knowledge_base().delete_document(filename):
        raise HTTPException(status_code=404, detail="Document not found.")

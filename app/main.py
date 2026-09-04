import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

from .api.deps import get_knowledge_base, ingestion_tasks
from .api.routes import chat, documents, health, query
from .config import get_settings
from .core.logging import RequestLoggingMiddleware, configure_logging

__all__ = ["app", "create_app", "get_knowledge_base", "ingestion_tasks"]


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(title="Enterprise Knowledge AI Assistant", version="0.1.0")
    application.add_middleware(RequestLoggingMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip() for origin in get_settings().cors_origins.split(",") if origin.strip()
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID", "X-User-ID"],
    )

    @application.exception_handler(Exception)
    async def unhandled_exception(request: Request, error: Exception) -> JSONResponse:
        logging.getLogger(__name__).exception(
            "unhandled_request_error", extra={"path": request.url.path}
        )
        return JSONResponse(status_code=500, content={"detail": "An internal server error occurred."})

    @application.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")

    application.include_router(health.router)
    application.include_router(documents.router)
    application.include_router(query.router)
    application.include_router(chat.router)
    return application


app = create_app()

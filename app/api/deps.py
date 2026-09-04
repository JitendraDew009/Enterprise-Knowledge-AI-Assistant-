from functools import lru_cache

from ..config import get_settings
from ..pg_rag import PgKnowledgeBase
from ..rag import KnowledgeBase

ingestion_tasks: dict[str, dict[str, int | str]] = {}


@lru_cache
def get_knowledge_base() -> KnowledgeBase:
    settings = get_settings()
    if settings.vector_backend.lower() == "pgvector":
        return PgKnowledgeBase(settings)
    return KnowledgeBase(settings)

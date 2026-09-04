import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import Settings
from app.core import security
from app.core.logging import JsonLogFormatter
from app.db.models import Conversation, Document, DocumentChunk, Message
from app.db.session import Base


def test_database_models_define_expected_tables() -> None:
    tables = set(Base.metadata.tables)
    assert tables == {"documents", "document_chunks", "conversations", "messages"}
    assert Document.__tablename__ == "documents"
    assert DocumentChunk.__tablename__ == "document_chunks"
    assert Conversation.__tablename__ == "conversations"
    assert Message.__tablename__ == "messages"


def test_database_models_track_document_and_chunk_metadata() -> None:
    document_columns = {column.name for column in Document.__table__.columns}
    chunk_columns = {column.name for column in DocumentChunk.__table__.columns}
    assert {"content_type", "size_bytes", "owner_id", "error_message", "updated_at"} <= document_columns
    assert {"extra_metadata", "created_at"} <= chunk_columns


def test_database_configuration_is_environment_driven(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./data/test.db")
    settings = Settings()
    assert settings.database_url == "sqlite:///./data/test.db"


def test_configured_api_key_rejects_invalid_credentials(monkeypatch) -> None:
    monkeypatch.setattr(security, "get_settings", lambda: Settings(application_api_key="expected"))
    with pytest.raises(HTTPException) as error:
        security.require_user(user_id="user-1", api_key="wrong")
    assert error.value.status_code == 401


def test_vector_backend_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    assert Settings().vector_backend == "pgvector"


def test_pgvector_is_default_backend() -> None:
    assert Settings(_env_file=None).vector_backend == "pgvector"


def test_invalid_foundation_settings_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=200, chunk_overlap=200)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="LOUD")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vector_backend="pgvector", embedding_dimensions=384)


def test_json_log_formatter_preserves_structured_context() -> None:
    import json
    import logging

    record = logging.LogRecord(
        name="atlas.request",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request_complete",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-1"
    record.duration_ms = 12.5

    payload = json.loads(JsonLogFormatter().format(record))
    assert payload["message"] == "request_complete"
    assert payload["request_id"] == "request-1"
    assert payload["duration_ms"] == 12.5

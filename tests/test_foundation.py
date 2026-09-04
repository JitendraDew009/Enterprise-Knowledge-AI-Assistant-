import pytest
from fastapi import HTTPException

from app.config import Settings
from app.core import security
from app.db.models import Conversation, Document, DocumentChunk, Message
from app.db.session import Base


def test_database_models_define_expected_tables() -> None:
    tables = set(Base.metadata.tables)
    assert tables == {"documents", "document_chunks", "conversations", "messages"}
    assert Document.__tablename__ == "documents"
    assert DocumentChunk.__tablename__ == "document_chunks"
    assert Conversation.__tablename__ == "conversations"
    assert Message.__tablename__ == "messages"


def test_database_configuration_is_environment_driven(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./data/test.db")
    settings = Settings()
    assert settings.database_url == "sqlite:///./data/test.db"


def test_configured_api_key_rejects_invalid_credentials(monkeypatch) -> None:
    monkeypatch.setattr(security, "get_settings", lambda: Settings(application_api_key="expected"))
    with pytest.raises(HTTPException) as error:
        security.require_user(user_id="user-1", api_key="wrong")
    assert error.value.status_code == 401

from app.config import Settings
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

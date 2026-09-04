from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.api import deps
from app.config import Settings
from app.db.models import Base
from app.db.session import get_db
from app.rag import KnowledgeBase


def test_document_workflow(tmp_path, monkeypatch) -> None:
    knowledge_base = KnowledgeBase(
        Settings(
            chroma_persist_directory=str(tmp_path / "chroma"), chunk_size=100, openai_api_key=""
        )
    )
    monkeypatch.setattr(deps, "get_knowledge_base", lambda: knowledge_base)
    client = TestClient(main.app)

    upload = client.post(
        "/documents",
        files={"file": ("policy.md", b"Remote work is supported three days per week.", "text/markdown")},
    )
    assert upload.status_code == 200
    assert upload.json()["chunks_indexed"] == 1

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert listed.json() == [{"filename": "policy.md", "chunks": 1}]

    query = client.post("/query", json={"question": "How often is remote work supported?"})
    assert query.status_code == 200
    assert query.json()["sources"][0]["source"] == "policy.md"

    deleted = client.delete("/documents/policy.md")
    assert deleted.status_code == 204
    assert client.get("/documents").json() == []
    assert client.delete("/documents/policy.md").status_code == 404


def test_chat_endpoint_persists_a_conversation(tmp_path, monkeypatch) -> None:
    knowledge_base = KnowledgeBase(
        Settings(
            chroma_persist_directory=str(tmp_path / "chroma"), chunk_size=100, openai_api_key=""
        )
    )
    knowledge_base.add_document("policy.md", "Remote work is supported three days per week.")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    def override_db():
        with session_factory() as session:
            yield session

    monkeypatch.setattr(deps, "get_knowledge_base", lambda: knowledge_base)
    main.app.dependency_overrides[get_db] = override_db
    try:
        response = TestClient(main.app).post(
            "/chat",
            headers={"X-User-ID": "user-1"},
            json={"question": "How often is remote work supported?"},
        )
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["conversation_id"]
    assert response.json()["sources"][0]["page"] == 1


def test_background_upload_returns_task_status(tmp_path, monkeypatch) -> None:
    knowledge_base = KnowledgeBase(
        Settings(
            chroma_persist_directory=str(tmp_path / "chroma"), chunk_size=100, openai_api_key=""
        )
    )
    monkeypatch.setattr(deps, "get_knowledge_base", lambda: knowledge_base)
    response = TestClient(main.app).post(
        "/documents?background=true",
        files={"file": ("policy.md", b"Remote work is supported.", "text/markdown")},
    )
    assert response.status_code == 202
    task = TestClient(main.app).get(f"/documents/tasks/{response.json()['task_id']}")
    assert task.status_code == 200
    assert task.json()["status"] in {"queued", "processing", "completed"}

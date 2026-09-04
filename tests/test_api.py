from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.rag import KnowledgeBase


def test_document_workflow(tmp_path, monkeypatch) -> None:
    knowledge_base = KnowledgeBase(
        Settings(chroma_persist_directory=str(tmp_path / "chroma"), chunk_size=100)
    )
    monkeypatch.setattr(main, "get_knowledge_base", lambda: knowledge_base)
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

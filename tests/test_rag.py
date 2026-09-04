from app.config import Settings
from app.rag import DeterministicEmbeddings, KnowledgeBase, read_supported_document


def test_text_documents_are_decoded() -> None:
    assert read_supported_document("policy.md", b"Remote work is supported.") == "Remote work is supported."


def test_unsupported_documents_are_rejected() -> None:
    try:
        read_supported_document("policy.docx", b"content")
    except ValueError as error:
        assert "Unsupported file type" in str(error)
    else:
        raise AssertionError("Expected unsupported file type error")


def test_local_embeddings_are_deterministic() -> None:
    embeddings = DeterministicEmbeddings()
    assert embeddings.embed_query("Remote work policy") == embeddings.embed_query("Remote work policy")


def test_documents_can_be_indexed_and_retrieved(tmp_path) -> None:
    knowledge_base = KnowledgeBase(
        Settings(chroma_persist_directory=str(tmp_path / "chroma"), chunk_size=100)
    )
    assert knowledge_base.add_document("policy.md", "Remote work is supported three days per week.") == 1
    sources = knowledge_base.retrieve("How often is remote work supported?", limit=1)
    assert len(sources) == 1
    assert sources[0].source == "policy.md"

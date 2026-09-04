from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from app.db.models import Document, DocumentChunk
from app.providers.embeddings import DeterministicEmbeddings


def test_embedding_provider_returns_stable_vectors() -> None:
    provider = DeterministicEmbeddings(size=16)
    first = provider.embed_query("remote work policy")
    assert first == provider.embed_query("remote work policy")
    assert len(first) == 16


def test_pgvector_retrieval_compiles_cosine_distance_query() -> None:
    distance = DocumentChunk.embedding.cosine_distance([0.0] * 1536).label("distance")
    statement = (
        select(DocumentChunk, Document, distance)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.status == "ready")
        .order_by(distance)
        .limit(4)
    )
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "<=>" in sql
    assert "ORDER BY distance" in sql

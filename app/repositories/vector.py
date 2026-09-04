from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Document, DocumentChunk


class PgVectorRepository:
    def add_chunks(
        self,
        session: Session,
        document: Document,
        chunks: list[tuple[str, int, int | None, list[float]]],
    ) -> None:
        session.add_all(
            [
                DocumentChunk(
                    document_id=document.id,
                    content=content,
                    chunk_index=chunk_index,
                    page_number=page_number,
                    embedding=embedding,
                )
                for content, chunk_index, page_number, embedding in chunks
            ]
        )
        session.flush()

    def search(
        self,
        session: Session,
        query_embedding: list[float],
        limit: int,
    ) -> list[tuple[DocumentChunk, Document, float]]:
        distance = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
        statement = (
            select(DocumentChunk, Document, distance)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == "ready")
            .order_by(distance)
            .limit(limit)
        )
        return [(chunk, document, float(distance_value)) for chunk, document, distance_value in session.execute(statement)]

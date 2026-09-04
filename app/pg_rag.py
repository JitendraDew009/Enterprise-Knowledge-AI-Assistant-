from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .db.models import Document
from .db.session import SessionLocal
from .ingestion import DocumentPage, chunk_document
from .providers.embeddings import build_embedding_provider
from .rag import DocumentSummary, KnowledgeBase, RetrievedSource
from .repositories.vector import PgVectorRepository


class PgKnowledgeBase(KnowledgeBase):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings: Embeddings = build_embedding_provider(
            settings, size=settings.embedding_dimensions
        )
        self.repository = PgVectorRepository()
        overlap = min(settings.chunk_overlap, max(settings.chunk_size - 1, 0))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=overlap,
        )

    def add_document_pages(self, filename: str, pages: list[DocumentPage]) -> int:
        documents = chunk_document(filename, pages, self.splitter)
        if not documents:
            return 0
        embeddings = self.embeddings.embed_documents([document.page_content for document in documents])
        with SessionLocal() as session:
            previous = session.query(Document).filter(Document.filename == filename).first()
            if previous:
                session.delete(previous)
                session.flush()
            document = Document(filename=filename, status="processing")
            session.add(document)
            session.flush()
            self.repository.add_chunks(
                session,
                document,
                [
                    (
                        item.page_content,
                        int(item.metadata["chunk"]),
                        int(item.metadata["page"]),
                        vector,
                    )
                    for item, vector in zip(documents, embeddings, strict=True)
                ],
            )
            document.status = "ready"
            session.commit()
        return len(documents)

    def list_documents(self) -> list[DocumentSummary]:
        with SessionLocal() as session:
            documents = session.query(Document).filter(Document.status == "ready").all()
            return [DocumentSummary(filename=item.filename, chunks=len(item.chunks)) for item in documents]

    def delete_document(self, filename: str) -> bool:
        with SessionLocal() as session:
            document = session.query(Document).filter(Document.filename == filename).first()
            if not document:
                return False
            session.delete(document)
            session.commit()
            return True

    def retrieve(self, question: str, limit: int | None = None) -> list[RetrievedSource]:
        query_embedding = self.embeddings.embed_query(question)
        with SessionLocal() as session:
            matches = self.repository.search(session, query_embedding, limit or self.settings.retrieval_k)
            return [
                RetrievedSource(
                    source=document.filename,
                    excerpt=chunk.content,
                    score=round(1 / (1 + distance), 4),
                    page=chunk.page_number,
                    chunk=chunk.chunk_index,
                )
                for chunk, document, distance in matches
            ]

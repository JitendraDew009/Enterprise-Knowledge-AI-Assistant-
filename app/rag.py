import logging
import re
from dataclasses import dataclass
from typing import ClassVar

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .ingestion import DocumentPage, chunk_document, extract_document_pages
from .providers.embeddings import DeterministicEmbeddings, build_embedding_provider
from .providers.generation import build_grounded_prompt

__all__ = ["DeterministicEmbeddings", "KnowledgeBase", "read_supported_document"]


@dataclass
class RetrievedSource:
    source: str
    excerpt: str
    score: float | None
    page: int | None = None
    chunk: int | None = None


@dataclass
class DocumentSummary:
    filename: str
    chunks: int


class KnowledgeBase:
    _STOP_WORDS: ClassVar[set[str]] = {
        "a", "an", "and", "are", "can", "do", "for", "how", "i", "in", "is",
        "it", "of", "on", "or", "our", "the", "their", "this", "to", "what", "with",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings = self._build_embeddings()
        self.store = Chroma(
            collection_name="enterprise_documents",
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_directory,
        )
        overlap = min(settings.chunk_overlap, max(settings.chunk_size - 1, 0))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=overlap,
        )

    def _build_embeddings(self) -> Embeddings:
        return build_embedding_provider(self.settings)

    def add_document(self, filename: str, content: str) -> int:
        return self.add_document_pages(filename, [DocumentPage(1, content)])

    def add_document_pages(self, filename: str, pages: list[DocumentPage]) -> int:
        chunks = chunk_document(filename, pages, self.splitter)
        if not chunks:
            return 0
        self.delete_document(filename)
        self.store.add_documents(chunks)
        return len(chunks)

    def list_documents(self) -> list[DocumentSummary]:
        metadata = self.store.get(include=["metadatas"]).get("metadatas", [])
        counts: dict[str, int] = {}
        for item in metadata:
            source = str((item or {}).get("source", "unknown"))
            counts[source] = counts.get(source, 0) + 1
        return [
            DocumentSummary(filename=filename, chunks=counts[filename])
            for filename in sorted(counts)
        ]

    def delete_document(self, filename: str) -> bool:
        documents = self.store.get(where={"source": filename}, include=["metadatas"])
        if not documents.get("ids"):
            return False
        self.store.delete(where={"source": filename})
        return True

    def retrieve(self, question: str, limit: int | None = None) -> list[RetrievedSource]:
        matches = self.store.similarity_search_with_score(
            question, k=limit or self.settings.retrieval_k
        )
        sources = [
            RetrievedSource(
                source=str(document.metadata.get("source", "unknown")),
                excerpt=document.page_content,
                score=round(1 / (1 + float(score)), 4),
                page=document.metadata.get("page"),
                chunk=document.metadata.get("chunk"),
            )
            for document, score in matches
        ]
        return [
            source
            for source in sources
            if source.score is not None and source.score >= self.settings.retrieval_score_threshold
        ]

    @staticmethod
    def _has_shared_terms(question: str, sources: list[RetrievedSource]) -> bool:
        question_terms = set(re.findall(r"[a-z0-9]+", question.lower())) - KnowledgeBase._STOP_WORDS
        document_terms = set(
            re.findall(r"[a-z0-9]+", " ".join(source.excerpt for source in sources).lower())
        ) - KnowledgeBase._STOP_WORDS
        return bool(question_terms & document_terms)

    @staticmethod
    def _extractive_answer(question: str, sources: list[RetrievedSource]) -> str:
        question_terms = set(re.findall(r"[a-z0-9]+", question.lower())) - KnowledgeBase._STOP_WORDS
        candidates = re.split(r"(?<=[.!?])\s+", " ".join(source.excerpt for source in sources))
        ranked = sorted(
            candidates,
            key=lambda sentence: len(
                question_terms & set(re.findall(r"[a-z0-9]+", sentence.lower()))
            ),
            reverse=True,
        )
        answer = " ".join(sentence.strip() for sentence in ranked[:2] if sentence.strip())
        return answer[:600].rstrip() + ("..." if len(answer) > 600 else "")

    def answer(self, question: str) -> tuple[str, list[RetrievedSource]]:
        sources = self.retrieve(question)
        if not sources or not self._has_shared_terms(question, sources):
            return "I could not find relevant information in the uploaded documents.", []

        context = "\n\n".join(
            f"[source={source.source} page={source.page or 'unknown'} chunk={source.chunk}]\n"
            f"{source.excerpt}"
            for source in sources
        )
        if not self.settings.openai_api_key:
            return self._extractive_answer(question, sources), sources

        try:
            response = (
                build_grounded_prompt()
                | ChatOpenAI(
                    api_key=self.settings.openai_api_key,
                    model=self.settings.openai_chat_model,
                    temperature=0,
                    timeout=self.settings.llm_timeout_seconds,
                    max_retries=self.settings.llm_max_retries,
                )
            ).invoke({"question": question, "context": context})
            return str(response.content), sources
        except Exception:
            logging.getLogger(__name__).exception("grounded_generation_failed")
            return self._extractive_answer(question, sources), sources


def read_supported_document(filename: str, content: bytes) -> str:
    return "\n\n".join(page.text for page in extract_document_pages(filename, content))

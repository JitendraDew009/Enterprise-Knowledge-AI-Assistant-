import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


@dataclass
class RetrievedSource:
    source: str
    excerpt: str
    score: float | None


@dataclass
class DocumentSummary:
    filename: str
    chunks: int


class DeterministicEmbeddings(Embeddings):
    def __init__(self, size: int = 384) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.size
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        magnitude = sum(value * value for value in vector) ** 0.5
        return [value / magnitude for value in vector] if magnitude else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class KnowledgeBase:
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
        if self.settings.openai_api_key:
            return OpenAIEmbeddings(
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_embedding_model,
            )
        return DeterministicEmbeddings()

    def add_document(self, filename: str, content: str) -> int:
        chunks = self.splitter.create_documents(
            [content], metadatas=[{"source": filename}]
        )
        if not chunks:
            return 0
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
        return [
            RetrievedSource(
                source=str(document.metadata.get("source", "unknown")),
                excerpt=document.page_content,
                score=round(1 / (1 + float(score)), 4),
            )
            for document, score in matches
        ]

    def answer(self, question: str) -> tuple[str, list[RetrievedSource]]:
        sources = self.retrieve(question)
        if not sources:
            return "I could not find relevant information in the uploaded documents.", []

        context = "\n\n".join(
            f"[{source.source}]\n{source.excerpt}" for source in sources
        )
        if not self.settings.openai_api_key:
            return (
                "I found these relevant passages:\n\n"
                + "\n\n".join(f"- {source.excerpt}" for source in sources),
                sources,
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Answer only from the supplied context. If the context does not contain "
                        "the answer, say so. Cite sources inline using [filename]."
                    ),
                ),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ]
        )
        response = (prompt | ChatOpenAI(
            api_key=self.settings.openai_api_key,
            model=self.settings.openai_chat_model,
            temperature=0,
        )).invoke({"question": question, "context": context})
        return str(response.content), sources


def read_supported_document(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
    raise ValueError("Unsupported file type. Upload a .txt, .md, or .pdf file.")

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
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
        return FakeEmbeddings(size=384)

    def add_document(self, filename: str, content: str) -> int:
        chunks = self.splitter.create_documents(
            [content], metadatas=[{"source": filename}]
        )
        if not chunks:
            return 0
        self.store.add_documents(chunks)
        return len(chunks)

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

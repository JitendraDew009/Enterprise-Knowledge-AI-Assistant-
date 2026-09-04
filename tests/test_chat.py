from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.models import Base, Message
from app.rag import RetrievedSource
from app.schemas.chat import ChatRequest
from app.services.chat import ConversationNotFoundError, ConversationService


class FakeKnowledgeBase:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def answer(self, question: str) -> tuple[str, list[RetrievedSource]]:
        self.questions.append(question)
        return "Grounded answer", [RetrievedSource("policy.md", "Policy text", 0.9)]


def test_chat_creates_and_continues_conversation() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    knowledge_base = FakeKnowledgeBase()
    service = ConversationService(Settings(openai_api_key=""), knowledge_base, session)

    first = service.chat(ChatRequest(question="What is the policy?"), "user-1")
    second = service.chat(
        ChatRequest(question="Can you summarize that?", conversation_id=first.conversation_id),
        "user-1",
    )

    assert first.conversation_id == second.conversation_id
    assert session.query(Message).count() == 4
    assert "user: What is the policy?" in knowledge_base.questions[1]


def test_chat_rejects_other_owner() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    service = ConversationService(Settings(openai_api_key=""), FakeKnowledgeBase(), session)

    response = service.chat(ChatRequest(question="Question"), "user-1")
    try:
        service.chat(
            ChatRequest(question="Follow up", conversation_id=response.conversation_id),
            "user-2",
        )
    except ConversationNotFoundError:
        return
    raise AssertionError("Expected an owner mismatch to be rejected")

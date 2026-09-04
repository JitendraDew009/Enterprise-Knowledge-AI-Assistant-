from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..db.models import Conversation, Message
from ..rag import KnowledgeBase
from ..schemas.chat import ChatRequest, ChatResponse, ChatSource


class ConversationNotFoundError(Exception):
    pass


class ConversationService:
    def __init__(self, settings: Settings, knowledge_base: KnowledgeBase, session: Session) -> None:
        self.settings = settings
        self.knowledge_base = knowledge_base
        self.session = session

    def chat(self, request: ChatRequest, owner_id: str) -> ChatResponse:
        conversation = self._get_or_create_conversation(request.conversation_id, owner_id)
        history = self._recent_history(conversation.id)
        question = self._contextual_question(request.question, history)
        answer, sources = self.knowledge_base.answer(question)

        self.session.add(Message(conversation_id=conversation.id, role="user", content=request.question))
        self.session.add(Message(conversation_id=conversation.id, role="assistant", content=answer))
        self.session.commit()
        return ChatResponse(
            conversation_id=conversation.id,
            answer=answer,
            sources=[ChatSource(**source.__dict__) for source in sources],
        )

    def _get_or_create_conversation(self, conversation_id: str | None, owner_id: str) -> Conversation:
        if conversation_id:
            conversation = self.session.get(Conversation, conversation_id)
            if not conversation or conversation.owner_id != owner_id:
                raise ConversationNotFoundError
            return conversation
        conversation = Conversation(owner_id=owner_id)
        self.session.add(conversation)
        self.session.flush()
        return conversation

    def _recent_history(self, conversation_id: str) -> list[Message]:
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(self.settings.conversation_history_limit)
        )
        return list(reversed(self.session.scalars(statement).all()))

    @staticmethod
    def _contextual_question(question: str, history: list[Message]) -> str:
        if not history:
            return question
        previous = "\n".join(f"{message.role}: {message.content}" for message in history)
        return f"Conversation context:\n{previous}\n\nCurrent question: {question}"

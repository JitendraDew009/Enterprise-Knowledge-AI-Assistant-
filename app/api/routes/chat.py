from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...config import get_settings
from ...core.security import require_user
from ...db.session import get_db
from ...schemas.chat import ChatRequest, ChatResponse
from ...services.chat import ConversationNotFoundError, ConversationService
from .. import deps

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    session: Annotated[Session, Depends(get_db)],
    user_id: Annotated[str, Depends(require_user)],
) -> ChatResponse:
    try:
        return ConversationService(get_settings(), deps.get_knowledge_base(), session).chat(
            request, user_id
        )
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail="Conversation not found.") from error

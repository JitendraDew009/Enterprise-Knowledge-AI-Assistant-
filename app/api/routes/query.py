from typing import Annotated

from fastapi import APIRouter, Depends

from ...core.security import require_user
from ...schemas.query import QueryRequest, QueryResponse, SourceResponse
from .. import deps

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_knowledge_base(
    request: QueryRequest,
    _user_id: Annotated[str, Depends(require_user)],
) -> QueryResponse:
    answer, sources = deps.get_knowledge_base().answer(request.question)
    return QueryResponse(answer=answer, sources=[SourceResponse(**source.__dict__) for source in sources])

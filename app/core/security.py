import secrets

from fastapi import Header, HTTPException

from ..config import get_settings


def require_user(
    user_id: str | None = Header(default=None, alias="X-User-ID"),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    settings = get_settings()
    if settings.application_api_key and not secrets.compare_digest(
        api_key or "", settings.application_api_key
    ):
        raise HTTPException(status_code=401, detail="Valid API credentials are required.")
    normalized_user_id = (user_id or "anonymous").strip()
    if not normalized_user_id or len(normalized_user_id) > 128:
        raise HTTPException(status_code=400, detail="X-User-ID must be 1-128 characters.")
    return normalized_user_id

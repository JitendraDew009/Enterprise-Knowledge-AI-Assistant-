import secrets

import jwt
from fastapi import Header, HTTPException

from ..config import get_settings


def require_user(
    user_id: str | None = Header(default=None, alias="X-User-ID"),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> str:
    settings = get_settings()
    authorization = authorization if isinstance(authorization, str) else None
    api_key = api_key if isinstance(api_key, str) else None
    user_id = user_id if isinstance(user_id, str) else None
    if settings.jwt_secret:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="A Bearer token is required.")
        try:
            claims = jwt.decode(
                authorization.removeprefix("Bearer "),
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm],
                options={"require": ["sub"]},
            )
        except jwt.PyJWTError as error:
            raise HTTPException(status_code=401, detail="Invalid authentication token.") from error
        user_id = str(claims["sub"])
    if settings.application_api_key and not secrets.compare_digest(
        api_key or "", settings.application_api_key
    ):
        raise HTTPException(status_code=401, detail="Valid API credentials are required.")
    normalized_user_id = (user_id or "anonymous").strip()
    if not normalized_user_id or len(normalized_user_id) > 128:
        raise HTTPException(status_code=400, detail="X-User-ID must be 1-128 characters.")
    return normalized_user_id

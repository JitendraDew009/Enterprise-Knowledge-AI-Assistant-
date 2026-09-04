import jwt
import pytest
from fastapi import HTTPException

from app.config import Settings
from app.core import security


@pytest.fixture
def jwt_settings(monkeypatch):
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: Settings(jwt_secret="test-secret-that-is-at-least-32-bytes-long"),
    )


def test_valid_jwt_returns_subject(jwt_settings) -> None:
    token = jwt.encode(
        {"sub": "user-7"}, "test-secret-that-is-at-least-32-bytes-long", algorithm="HS256"
    )
    assert security.require_user(authorization=f"Bearer {token}") == "user-7"


def test_missing_jwt_is_rejected(jwt_settings) -> None:
    with pytest.raises(HTTPException, match="Bearer token") as error:
        security.require_user()
    assert error.value.status_code == 401


def test_invalid_jwt_is_rejected(jwt_settings) -> None:
    with pytest.raises(HTTPException, match="Invalid authentication token") as error:
        security.require_user(authorization="Bearer invalid")
    assert error.value.status_code == 401

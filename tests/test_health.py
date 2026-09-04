from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_returns_ok_when_db_available(monkeypatch) -> None:
    monkeypatch.setattr("app.api.routes.health.check_database", lambda: True)
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_readiness_returns_503_when_db_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("app.api.routes.health.check_database", lambda: False)
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "unavailable"}

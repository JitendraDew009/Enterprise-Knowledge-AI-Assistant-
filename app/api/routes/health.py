from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ...db.session import check_database

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
def readiness() -> dict[str, str]:
    database_status = "ok" if check_database() else "unavailable"
    overall_status = "ok" if database_status == "ok" else "degraded"
    payload = {"status": overall_status, "database": database_status}
    if overall_status != "ok":
        return JSONResponse(status_code=503, content=payload)
    return payload

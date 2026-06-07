"""Health-check endpoint confirming the service is running."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health_check() -> JSONResponse:
    """Return a simple liveness signal.

    Returns:
        JSON body with status and service name.
    """
    return JSONResponse({"status": "ok", "service": "LUFY"})

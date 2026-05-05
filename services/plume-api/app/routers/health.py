"""Health + version endpoints."""

from fastapi import APIRouter

from ..config import settings

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/version")
def version() -> dict:
    return {
        "model_version": settings.model_version,
        "grid": f"{settings.grid_ny}x{settings.grid_nx}x{settings.grid_nz}",
        "domain_km": settings.domain_km,
    }

"""POST /v1/forecast — wind/T/FWI at a single point.

Returns values sampled at the domain center (point forecast). The 3D box
is also cached under a job id (future: stream it back for the viewer).
"""

from __future__ import annotations

import struct
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..config import settings
from ..data.domain import build_and_infer
from ..inference.postprocessor import extract_level
from ..middleware.rate_limit import rate_limit_dep

# Optional FWI import — fall back gracefully if shared package is not on path.
try:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3].parent))
    from shared.fwi import compute_fwi_field  # type: ignore
    _HAS_FWI = True
except Exception:
    compute_fwi_field = None  # type: ignore
    _HAS_FWI = False

router = APIRouter(prefix="/v1")


class ForecastRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: datetime
    variables: list[str] = Field(
        default_factory=lambda: ["wind_speed_60m", "wind_direction_60m"]
    )


class ForecastResponse(BaseModel):
    location: dict
    timestamp: str
    wind_speed_60m_ms: float | None = None
    wind_direction_60m_deg: float | None = None
    temperature_60m_K: float | None = None
    fwi: float | None = None
    model_version: str
    cache_hit: bool = False


def _point_center(field_2d: np.ndarray) -> float:
    ny, nx = field_2d.shape
    return float(field_2d[ny // 2, nx // 2])


@router.post("/forecast", response_model=ForecastResponse, dependencies=[Depends(rate_limit_dep)])
def forecast(req: ForecastRequest, request: Request) -> ForecastResponse:
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="inference engine not loaded on this instance",
        )

    try:
        result = build_and_infer(req.latitude, req.longitude, req.timestamp, engine)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    z = result.z_levels
    target_z = 60.0
    u60 = extract_level(result.u, z, target_z)
    v60 = extract_level(result.v, z, target_z)
    T60 = extract_level(result.T, z, target_z)

    u_center = _point_center(u60)
    v_center = _point_center(v60)
    T_center = _point_center(T60)
    speed = float(np.hypot(u_center, v_center))
    # Meteorological direction: direction wind comes FROM, 0=N, clockwise
    direction = float((np.rad2deg(np.arctan2(-u_center, -v_center)) + 360.0) % 360.0)

    # Elevation at center
    terr = result.terrain
    elev_center = float(terr[terr.shape[0] // 2, terr.shape[1] // 2])

    fwi_value: float | None = None
    if _HAS_FWI and "fwi" in req.variables:
        try:
            q60 = extract_level(result.q, z, target_z)
            q_center = _point_center(q60)
            # Instantaneous FWI at the domain center. No rain series available
            # here, so use rain=0 and default carry-over values — this yields a
            # "fair weather, climatology" FWI, acceptable for a prototype.
            out = compute_fwi_field(
                t_kelvin=np.array([T_center]),
                q_kgkg=np.array([max(q_center, 1e-6)]),
                p_hpa=np.array([1013.0]),
                u_ms=np.array([u_center]),
                v_ms=np.array([v_center]),
                rain_mm=np.array([0.0]),
                month=req.timestamp.month,
            )
            fwi_value = float(out["fwi"][0])
        except Exception as e:
            print(f"[plume] FWI computation failed: {e}")

    resp = ForecastResponse(
        location={"lat": req.latitude, "lon": req.longitude, "elevation_m": elev_center},
        timestamp=req.timestamp.isoformat(),
        wind_speed_60m_ms=speed,
        wind_direction_60m_deg=direction,
        temperature_60m_K=T_center,
        fwi=fwi_value,
        model_version=settings.model_version,
        cache_hit=False,
    )
    return resp


@router.get("/box.bin", dependencies=[Depends(rate_limit_dep)])
def box_bin(
    request: Request,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    timestamp: datetime = Query(...),
) -> Response:
    """Stream the full 3D box in the PLM2 binary format consumed by the viewer.

    This lets the frontend run "dynamic" mode: user picks a location, hits this
    endpoint, and the same loader.js that reads demo files handles the result.
    """
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="inference engine not loaded")

    try:
        result = build_and_infer(lat, lon, timestamp, engine)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    ny, nx = result.terrain.shape
    nz = result.u.shape[2]
    dx = (settings.domain_km * 1000.0) / nx
    dy = (settings.domain_km * 1000.0) / ny

    header = b"PLM2" + struct.pack(
        "<IIIffffxxxxxxxx", nx, ny, nz, float(dx), float(dy), float(lat), float(lon)
    )
    assert len(header) == 40

    buf = bytearray()
    buf += header
    buf += result.z_levels.astype(np.float32).tobytes()
    buf += result.terrain.astype(np.float32).tobytes()
    buf += result.u.astype(np.float32).tobytes()
    buf += result.v.astype(np.float32).tobytes()
    buf += result.w.astype(np.float32).tobytes()

    return Response(content=bytes(buf), media_type="application/octet-stream")

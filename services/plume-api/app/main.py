"""Plume API — FastAPI entry point.

Serves:
  /                → landing page + 3D viewer
  /static/*        → JS/CSS/assets
  /demo/<name>.bin → pre-computed demo cases (served by StaticFiles)
  /v1/forecast     → POST, returns downscaled wind/T/FWI
  /health          → GET
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .inference.engine import FNOEngine
from .routers import forecast, health

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
DEMO_DIR = APP_DIR / "demo_data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup if present; otherwise run in "viewer only" mode.
    if settings.model_path.exists():
        try:
            app.state.engine = FNOEngine(settings.model_path)
            print(f"[plume] loaded model from {settings.model_path}")
        except Exception as e:
            print(f"[plume] WARNING: model load failed: {e}")
            app.state.engine = None
    else:
        print(f"[plume] no model at {settings.model_path} — viewer-only mode")
        app.state.engine = None
    yield


app = FastAPI(
    title="Plume API",
    description="Wind downscaling from ERA5 25 km to 30 m via CFD surrogate (FNO3D).",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(forecast.router)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/demo", StaticFiles(directory=str(DEMO_DIR)), name="demo")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

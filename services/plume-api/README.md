# Plume API

Wind downscaling API + 3D viewer showcase. Serves the FNO3D surrogate trained on
campaign_9k (8998 CFD cases, site-split, RMSE=0.564 m/s).

## Architecture

```
app/
├── main.py              FastAPI entry (serves /, /static, /v1/forecast)
├── config.py            Settings (model path, cache dir, rate limits)
├── routers/
│   ├── forecast.py      POST /v1/forecast (async job pattern)
│   └── health.py        GET /health
├── inference/
│   ├── engine.py        TorchScript model wrapper (CPU ARM)
│   ├── preprocessor.py  Build 7ch input from terrain + z0 + ERA5 profiles
│   └── postprocessor.py Denormalize, extract z=60m, compute FWI
├── data/
│   ├── terrain.py       COP-DEM 30m → 128×128 crop, cached on disk
│   ├── era5.py          Open-Meteo IFS fetch → 32-level AGL profile
│   ├── landcover.py     WorldCover z0 (or constant fallback)
│   └── domain.py        Assemble input volume for (lat, lon, timestamp)
├── middleware/
│   └── rate_limit.py    Token bucket (100 req/day per IP)
├── static/
│   ├── index.html       Landing + 3D viewer
│   ├── js/              Three.js viewer modules
│   └── css/
└── demo_data/           Pre-computed showcase cases (*.bin, gzip)
```

## Deployment

Target host: `enjoy@wx-outdoor.com` (OCI ARM A1 free tier, 4 OCPU, 24 GB RAM, no GPU)

**Important**: Do NOT write anything to `/data` on the OCI host (reserved).
Use `/home/enjoy/plume/` or `/opt/plume/`.

## Model

- FNO3D 33.6M params, input (1, 7, 128, 128, 32), output (1, 5, 128, 128, 32)
- Format: TorchScript (`fno3d_9k.ts.pt`, ~268 MB) — ONNX blocked by `fft_rfftn`
- Expected latency on OCI ARM: 2-4 seconds per inference (prototype, async OK)
- Normalization scales (identical to training):
  - terrain/500, z0/1, u/20, v/20, T/30, q/0.01, k/1

## Run locally

```bash
cd services/plume-api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
open http://localhost:8000
```

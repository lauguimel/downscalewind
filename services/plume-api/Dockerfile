# Plume API — ARM64 (OCI A1 Ampere target).
#
# Build:
#   docker buildx build --platform linux/arm64 -t plume-api:latest .
#
# Run:
#   docker run -p 8000:8000 -v /home/enjoy/plume/models:/app/models \
#              -v /home/enjoy/plume/cache:/app/cache \
#              plume-api:latest
#
# IMPORTANT: on the OCI host, do NOT mount or write to /data — reserved.

FROM --platform=linux/arm64 python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for rasterio (GDAL) + matplotlib-free torch CPU wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev \
        libexpat1 \
        libopenblas-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Use CPU-only torch to keep the image small on ARM
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app ./app

# Model + cache are mounted at runtime (not baked into the image to keep it small)
ENV PLUME_MODEL_PATH=/app/models/fno3d_9k.ts.pt \
    PLUME_CACHE_DIR=/app/cache \
    PLUME_DEMO_DATA_DIR=/app/app/demo_data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

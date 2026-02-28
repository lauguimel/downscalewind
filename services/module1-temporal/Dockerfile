FROM python:3.11-slim

# TODO : compléter quand le module est implémenté
# Dépendances prévues : torch, torch-geometric, torchdiffeq, zarr, mlflow

WORKDIR /app

COPY shared/ /app/shared/
RUN pip install --no-cache-dir -e /app/shared/

VOLUME ["/data", "/logs"]
ENV PYTHONPATH=/app

CMD ["echo", "Module 1 (temporal downscaling) — not yet implemented"]

FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour rasterio et NetCDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libnetcdf-dev \
    libhdf5-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Package partagé (installé en editable)
COPY shared/ /app/shared/
RUN pip install --no-cache-dir -e /app/shared/

# Dépendances du service
COPY services/data-ingestion/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code du service
COPY services/data-ingestion/ /app/

# Volumes attendus :
#   /data  → données brutes et traitées
#   /logs  → fichiers de log
VOLUME ["/data", "/logs"]

# Variables d'environnement
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# Commande par défaut (override via docker-compose)
CMD ["python", "ingest_era5.py", "--help"]

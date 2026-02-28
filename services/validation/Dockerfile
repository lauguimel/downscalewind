FROM python:3.11-slim

# TODO : compléter quand le module est implémenté
# Dépendances prévues : xarray, zarr, matplotlib, cartopy, seaborn, jupyterlab

WORKDIR /app

COPY shared/ /app/shared/
RUN pip install --no-cache-dir -e /app/shared/

EXPOSE 8888
VOLUME ["/data", "/logs"]
ENV PYTHONPATH=/app

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser",
     "--notebook-dir=/app/notebooks", "--ServerApp.token=''"]

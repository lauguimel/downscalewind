FROM openfoam/openfoam10-paraview510:latest

# TODO : finaliser quand le module est implémenté
# Base : OpenFOAM 10 (inclut buoyantSimpleFoam, snappyHexMesh)
# Ajouter : Python 3.11, fluidfoam, numpy-stl, rasterio, jinja2

USER root

# Python et dépendances Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# TODO : COPY requirements.txt et pip install

VOLUME ["/data", "/logs"]

# Source OpenFOAM dans le shell
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/openfoam10/etc/bashrc" >> /root/.bashrc

CMD ["echo", "Module 2A (CFD batch) — not yet implemented"]

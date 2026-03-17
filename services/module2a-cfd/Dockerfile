FROM opencfd/openfoam-default:2412

# Base: OpenFOAM ESI v2412 (simpleFoam, buoyantBoussinesqSimpleFoam, k-epsilon)
# Added: cfMesh (cartesianMesh), Python 3, fluidfoam, numpy-stl, rasterio, jinja2

USER root

# Build dependencies for cfMesh + Python/GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake flex \
    python3 python3-pip python3-dev \
    gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Compile cfMesh from ESI OpenFOAM modules repository
SHELL ["/bin/bash", "-c"]
RUN source /usr/lib/openfoam/openfoam2412/etc/bashrc && \
    mkdir -p $WM_PROJECT_DIR/modules && \
    cd $WM_PROJECT_DIR/modules && \
    git clone --depth 1 https://develop.openfoam.com/Community/integration-cfmesh.git cfmesh && \
    cd cfmesh && \
    ./Allwmake -j 2>&1 | tail -5

# Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Copy module code
COPY . /opt/module2a-cfd/
ENV PYTHONPATH="/opt/module2a-cfd:${PYTHONPATH}"

VOLUME ["/data", "/logs"]

# Source OpenFOAM in every bash invocation
RUN echo "source /usr/lib/openfoam/openfoam2412/etc/bashrc" >> /root/.bashrc

WORKDIR /case

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'module2a-cfd ready — use run_cfd_batch.py'"]

# terrainBlockMesher — Docker build on OpenFOAM 2.4.x
#
# Usage:
#   docker build -t terrainblockmesher:of24 .
#   docker run --rm -v /path/to/case:/case -w /case terrainblockmesher:of24
#
# The image compiles iwesol/terrainBlockMesher inside an OF 2.4.x container.
# polyMesh output is cross-version compatible with OF v2406+.
#
# Alternative base images if asaramet is unavailable:
#   alexisespinosa/openfoam:2.4.x
#   Or build from source (see Dockerfile.from-source)

FROM asaramet/openfoam-2.4.x:latest

USER root

# g++, make already present in image; wmake available after sourcing OF bashrc
# No apt-get needed (Debian Jessie repos are dead anyway)

# Copy iwesol sources
COPY iwesol/ /opt/iwesol/

ENV IWESOL=/opt/iwesol
ENV OF_BASHRC=/opt/openfoam/2.4.x/OpenFOAM-2.4.x/etc/bashrc

# Create unified source script
RUN echo '#!/bin/bash' > /opt/source-env.sh && \
    echo "source $OF_BASHRC" >> /opt/source-env.sh && \
    echo "export IWESOL=$IWESOL" >> /opt/source-env.sh && \
    echo "export OLDEV=\$IWESOL" >> /opt/source-env.sh && \
    echo 'source $IWESOL/OF/OF-2.3.1/etc/bashrc' >> /opt/source-env.sh && \
    echo 'source $IWESOL/nonOF/c++/etc/bashrc' >> /opt/source-env.sh && \
    chmod +x /opt/source-env.sh

# Create obj/ and dep/ directories for nonOF libs (required by Makefile)
RUN mkdir -p $IWESOL/nonOF/c++/src/io/obj $IWESOL/nonOF/c++/src/io/dep \
    && mkdir -p $IWESOL/nonOF/c++/src/data/obj $IWESOL/nonOF/c++/src/data/dep \
    && mkdir -p $IWESOL/nonOF/c++/src/plot/obj $IWESOL/nonOF/c++/src/plot/dep \
    && mkdir -p $IWESOL/nonOF/c++/lib

# Compile nonOF libraries first (pure C++, no OF dependency)
RUN bash -c 'source /opt/source-env.sh && \
    cd $IWESOL && \
    make lolCppIO && \
    make lolCppData && \
    make lolCppPlot'

# Compile OF libraries + application (uses wmake)
RUN bash -c 'source /opt/source-env.sh && \
    cd $IWESOL && \
    wmake libso OF/OF-2.3.1/src/basics && \
    wmake libso OF/OF-2.3.1/src/functions && \
    wmake libso OF/OF-2.3.1/src/mesh/blockMesh && \
    wmake libso OF/OF-2.3.1/src/geometry && \
    wmake libso OF/OF-2.3.1/src/mesh/meshTools && \
    wmake OF/OF-2.3.1/applications/utilities/mesh/terrainBlockMesher'

# Verify binary exists
RUN bash -c 'source /opt/source-env.sh && which terrainBlockMesher'

# Copy tutorials for easy testing
RUN cp -r $IWESOL/OF/OF-2.3.1/tutorials /opt/tutorials

ENTRYPOINT ["bash", "-c", "source /opt/source-env.sh && exec \"$@\"", "--"]
CMD ["terrainBlockMesher"]

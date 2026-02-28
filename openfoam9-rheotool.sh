#!/bin/bash

# Define constants
DOCKER_IMAGE="guiguitcho/openfoam9-rheotool:v1.4"
MOUNT_DIR=$(pwd)
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d | --dir    Directory to mount and use as home in the container (default: current directory)"
    echo "  -h | --help   Show this help message"
    echo ""
    exit 1
}
# macOS-specific setup for XQuartz
if [[ "$OSTYPE" == "darwin"* ]]; then
    DISPLAY_VAR="host.docker.internal:0"  # Default for macOS XQuartz
    echo "Allowing XQuartz connections from localhost..."
    xhost +localhost
else 
    DISPLAY_VAR=:0 # Default for Linux
fi

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -d|--dir)
            shift
            [ -d "$1" ] || { echo "Directory does not exist: $1"; exit 1; }
            MOUNT_DIR=$(cd "$1" && pwd)
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done


# Launch the container
echo "Launching Docker container with the following settings:"
echo "  Docker Image: $DOCKER_IMAGE"
echo "  Mounted Directory: $MOUNT_DIR"
echo "  Display: $DISPLAY_VAR"


docker run -it --rm \
    --platform linux/amd64 \
    -p 8888:8888 \
    -e DISPLAY="$DISPLAY_VAR" \
    -e LIBGL_ALWAYS_INDIRECT=1 \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$MOUNT_DIR":/data \
    -w /data \
    -e USER_ID="$USER_ID" \
    -e GROUP_ID="$GROUP_ID" \
    "$DOCKER_IMAGE" \
    /bin/bash
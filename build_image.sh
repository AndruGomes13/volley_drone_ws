#!/bin/bash
is_jetson() {
  if [[ -f /etc/nv_tegra_release ]]; then
    return 0
  fi
  return 1
}

if [[ "$(uname -s)" == "Darwin" ]]; then
  # macOS (Apple Silicon)
  BASE_IMAGE="ubuntu:20.04"
  IMAGE_NAME="ros_agilicious_mac:latest"
  echo "Building Image for macOS ARM64 with Ubuntu base image: $BASE_IMAGE"
elif is_jetson; then
  BASE_IMAGE="nvcr.io/nvidia/l4t-base:r35.1.0"
  IMAGE_NAME="ros_agilicious_jetson:latest"
  echo "Building Image with JetPack base image: $BASE_IMAGE"
else
  BASE_IMAGE="nvidia/cuda:12.2.0-base-ubuntu20.04"
  IMAGE_NAME="ros_agilicious_cuda:latest"
  echo "Building Image with CUDA base image: $BASE_IMAGE"
fi

sudo docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$IMAGE_NAME" \
  -f Dockerfile .
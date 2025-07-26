#!/usr/bin/env bash
set -euo pipefail

# Detect ROS installation prefix
if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/setup.bash" ]]; then
    # Likely on macOS with RoboStack
    ROS_PREFIX="$CONDA_PREFIX"
else
    # Default to Linux-style install
    ROS_PREFIX="/opt/ros/${ROS_DISTRO}"
fi

catkin config --init --mkdirs \
              --extend "$ROS_PREFIX" \
              --merge-devel \
              --cmake-args -DCMAKE_BUILD_TYPE=Release \
                            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
                            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
catkin build "$@"

./merge_compile_commands.py
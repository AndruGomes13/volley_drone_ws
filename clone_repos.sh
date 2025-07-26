#!/usr/bin/env bash
set -euo pipefail

# Usage: ./clone_workspace.sh [workspace_dir]
# Defaults to ~/ros_ws if no directory is provided.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${1:-$SCRIPT_DIR}"
SRC_DIR="$WORKSPACE_DIR/src"

echo $SRC_DIR


# Define the repositories to clone into the workspace src folder
# Replace <YOUR_RPG_REPO_URL> with the actual URL of your RPG repo
declare -A REPOS=(
  [rotors_simulator]="https://github.com/ethz-asl/rotors_simulator.git"
  [mav_comm]="https://github.com/ethz-asl/mav_comm.git"
  [eigen_catkin]="https://github.com/ethz-asl/eigen_catkin.git"
  [catkin_simple]="https://github.com/catkin/catkin_simple.git"
)

# Create the src directory if it doesn't exist
mkdir -p "$SRC_DIR"

echo "Cloning/updating repositories into $SRC_DIR"
for name in "${!REPOS[@]}"; do
  url="${REPOS[$name]}"
  dest="$SRC_DIR/$name"
  if [ ! -d "$dest" ]; then
    echo " - Cloning $name..."
    git clone "$url" "$dest"
  else
    echo " - Updating $name..."
    git -C "$dest" pull --ff-only
  fi
done

echo "All repositories are up to date in $SRC_DIR"
#!/usr/bin/env bash
set -euo pipefail
echo "[devcontainer] Preparing X11 auth…"

UID_NUM="$(id -u)"
XAUTH="/tmp/.docker.xauth-${UID_NUM}"
XSOCK="/tmp/.X11-unix"

# Make sure both mount points exist
mkdir -p "$XSOCK"          # directory is normally there, but be safe
touch   "$XAUTH"           # file MUST exist or the bind-mount fails
chmod 600 "$XAUTH"

# If we do have an X display, grab its cookie
if [[ -n ${DISPLAY:-} ]]; then
    xauth nlist "$DISPLAY" 2>/dev/null \
        | sed 's/^..../ffff/' \
        | xauth -f "$XAUTH" nmerge - 2>/dev/null || true
fi

# Export so devcontainer.json can see it
export XAUTHORITY="$XAUTH"
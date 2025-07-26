#!/usr/bin/env bash
set -euo pipefail

# Skip VNC server startup if DISPLAY_VNC is not set
if [[ -z "${DISPLAY_VNC:-}" ]]; then
    echo "No VNC display set, skipping VNC server startup"
    exec "$@"
fi


echo "Starting VNC server on display $DISPLAY_VNC"

: "${VNC_GEOM:=1980x1080x24}"
: "${VNC_PW:=}"
LOGDIR="/tmp"

VNC_PORT=$((5900 + ${DISPLAY_VNC#:}))

# ──────────────────────────
# X11 socket permissions
# ──────────────────────────
sudo mkdir -p /tmp/.X11-unix
sudo chown root:root /tmp/.X11-unix
sudo chmod 1777      /tmp/.X11-unix

# ──────────────────────────
# Optional VNC password
# ──────────────────────────
if [[ -n "$VNC_PW" && ! -f "$HOME/.vnc/passwd" ]]; then
     mkdir -p "$HOME/.vnc"
     echo "$VNC_PW" | vncpasswd -f > "$HOME/.vnc/passwd"
     chmod 600 "$HOME/.vnc/passwd"
fi

# ──────────────────────────
# Launch & DETACH each service
# ──────────────────────────
nohup Xvfb "${DISPLAY_VNC}" -screen 0 "${VNC_GEOM}" \
      >"${LOGDIR}/xvfb.log" 2>&1 &

nohup env DISPLAY="${DISPLAY_VNC}" openbox \
      >"${LOGDIR}/openbox.log" 2>&1 &

# Launch x11vnc in foreground (container lives as long as this runs)
nohup x11vnc \
      -display "${DISPLAY_VNC}" \
      -rfbport "${VNC_PORT}" \
      -forever -listen 0.0.0.0 \
      ${VNC_PW:+-rfbauth "${HOME}/.vnc/passwd"} \
      >"${LOGDIR}/x11vnc.log" 2>&1 &

disown -a

exec "$@"
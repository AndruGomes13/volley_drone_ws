#!/bin/sh
is_jetson() {
  if [ -f /etc/nv_tegra_release ]; then
    return 0
  fi
  return 1
}

is_mac() {
  [ "$(uname -s)" = "Darwin" ]
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKSPACE_DIR=${1:-"$SCRIPT_DIR"}

if is_mac; then
  # point at XQuartz over TCP
  DISPLAY_X11=host.docker.internal:0
  DISPLAY_VNC=:1
  export DISPLAY=${DISPLAY_VNC} # Set default DISPLAY with VNC
  ENV_DISPLAY="--env=DISPLAY_X11=${DISPLAY_X11} --env=DISPLAY_VNC=${DISPLAY_VNC} --env=DISPLAY=${DISPLAY}"

  # SSH agent forwarding
  SSH_OPTS="--volume /run/host-services/ssh-auth.sock:/ssh-agent \
  --env SSH_AUTH_SOCK=/ssh-agent"

else
  # keep the normal Linux display
  export DISPLAY=${DISPLAY:-:0}

  # SSH agent forwarding
  SSH_OPTS="--volume $SSH_AUTH_SOCK:/ssh-agent \
  --env SSH_AUTH_SOCK=/ssh-agent"
fi

if is_mac; then
  echo "Launching macOS + XQuartz mode"
  DISPLAY=${DISPLAY:-host.docker.internal:0}
  GPU_OPTIONS=""                # no NVIDIA on Mac
  HOSTNAME_OPTS=""              # drop --net=host / pid / ipc
  CAP_OPTS=""                   # drop special caps
  IMAGE_NAME="ros_agilicious_mac:latest"
elif is_jetson; then
  echo "Launching Jetson mode"
  DISPLAY=${DISPLAY:-:0}
  GPU_OPTIONS="--gpus all --runtime=nvidia"
  CAP_OPTS="--cap-add SYS_NICE --cap-add SYS_ADMIN --cap-add IPC_LOCK"
  IMAGE_NAME="ros_agilicious_jetson:latest"
else
  echo "Launching x86_64 + NVIDIA mode"
  DISPLAY=${DISPLAY:-:0}
  GPU_OPTIONS="--gpus all --runtime=nvidia"
  CAP_OPTS="--cap-add SYS_NICE --cap-add SYS_ADMIN --cap-add IPC_LOCK"
  IMAGE_NAME="ros_agilicious_cuda:latest"
fi

# Network options
if is_mac; then
  NET_OPTS="-p 11311:11311 \
            -p 5901:5901 \
            -p 5900:5900"     # expose ROS master
else
  NET_OPTS="--net=host --pid host --ipc host" # use host networking
fi



# Display related options
if [ ! is_mac ]; then 
  XAUTH=/tmp/.docker.xauth-${UID}
  XSOCK=/tmp/.X11-unix
  [ -e "$XAUTH" ] || touch "$XAUTH"
  chmod 600 "$XAUTH"       

  xauth nlist "$DISPLAY" \
    | sed 's/^..../ffff/' \
    | xauth -f "$XAUTH" nmerge - 2>/dev/null

  MOUNT_XSOCK="--volume "$XSOCK":"$XSOCK":rw"              # <--- don’t mount the host’s X socket
  MOUNT_XAUTH="--volume "$XAUTH":"$XAUTH":rw"
  ENV_XAUTHORITY="--env="XAUTHORITY=${XAUTH}""
else
  ENV_XAUTHORITY=""
  MOUNT_XSOCK=""
  MOUNT_XAUTH=""
fi

# Check nVidia GPU docker support
if ! is_mac && dpkg --get-selections \
     | grep -q "^nvidia-container-toolkit[[:space:]]*install" ; then
  echo "Starting docker with NVIDIA support"
else
  GPU_OPTIONS=""
fi
sudo docker run --privileged --rm -it \
  --volume "$WORKSPACE_DIR":/home/agilicious/catkin_ws:rw \
  ${MOUNT_XSOCK} \
  ${MOUNT_XAUTH} \
  ${SSH_OPTS} \
  --volume /dev:/dev:rw \
  --volume /var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket \
  ${NET_OPTS} \
  ${CAP_OPTS} \
  --shm-size=1gb \
  ${ENV_XAUTHORITY} \
  ${ENV_DISPLAY} \
  --env="QT_X11_NO_MITSHM=1" \
  --env="DISABLE_ROS1_EOL_WARNINGS=1" \
  --env="XDG_RUNTIME_DIR=/run/user/1000/" \
  --env="TERM=xterm-256color" \
  --env="HISTFILE=/home/agilicious/catkin_ws/mount/.zsh_history" \
  ${GPU_OPTIONS} \
  --ulimit rtprio=99 --ulimit rttime=-1 --ulimit memlock=-1 \
  --cpuset-cpus 2-5 \
  -u "agilicious" \
  ${IMAGE_NAME} \
  zsh

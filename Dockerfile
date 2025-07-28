ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04
FROM ${BASE_IMAGE} AS base

# ─── Build-time args  ─────────────────────
ARG ROS_DISTRO=noetic
ARG UBUNTU_CODENAME=focal
ARG USERNAME=agilicious
ARG UID=1000
ARG GID=1000
ARG VIDEO_GID=44

# ROS Installation
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl gnupg2 lsb-release ca-certificates sudo

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
| tee /etc/apt/sources.list.d/ros1.list > /dev/null

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ros-${ROS_DISTRO}-desktop-full

RUN sudo apt-get install -y \
  ros-${ROS_DISTRO}-octomap-msgs \
  ros-${ROS_DISTRO}-octomap-ros \
  ros-${ROS_DISTRO}-xacro \
  ros-${ROS_DISTRO}-plotjuggler-ros \
  ros-${ROS_DISTRO}-mavros \
  ros-${ROS_DISTRO}-mavros-extras


# 
ENV ROS_DISTRO=${ROS_DISTRO}
ENV HOME=/home/$USERNAME
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+${NVIDIA_DRIVER_CAPABILITIES},graphics}

# system deps, compilers, python, git, tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential software-properties-common \
      curl gnupg2 lsb-release sudo x11-apps xauth libgl1-mesa-glx libgl1-mesa-dri libglx-mesa0 mesa-utils \
      gcc-9 g++-9 clang-10 \
      python-is-python3 git nano wget htop \
      libyaml-cpp-dev libeigen3-dev libgoogle-glog-dev \
      ccache tmux net-tools iputils-ping usbutils screen \
      automake bison flex gperf libncurses5-dev libtool \
      libusb-1.0-0-dev pkg-config dfu-util \
      linux-tools-generic dbus xdg-utils clangd-18 geographiclib-tools && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 \
                        --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    rm -rf /var/lib/apt/lists/*
    
RUN geographiclib-get-geoids egm96-5
# ─── VNC / X11 desktop support ─────────────────────
RUN apt-get update  && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    xvfb fluxbox tigervnc-standalone-server tigervnc-common \
    openbox xvfb x11vnc openbox xfonts-base\ 
    websockify novnc \ 
    libgl1-mesa-dri libglx-mesa0 mesa-utils \  
    x11-utils x11-xserver-utils && \     
    rm -rf /var/lib/apt/lists/*

#--- Setup Python 3.8 ---
RUN apt-get update && apt-get install -y python3.8-venv python3.8-distutils
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8
RUN python3.8 -m pip install --no-cache-dir catkin-tools scipy typing_extensions

# --- Setup Python 3.11 Environment for inference ---
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    && apt-get clean

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN python3.11 -m pip install --upgrade pip setuptools wheel
RUN python3.11 -m pip install --upgrade --ignore-installed PyYAML "jax[cpu]" brax

# switch default compiler to clang-10
ENV CC=/usr/bin/clang-10
ENV CXX=/usr/bin/clang++-10

# create non-root user
RUN groupadd --gid ${GID} ${USERNAME} && \
    useradd --uid ${UID} --gid ${GID} --shell /bin/bash \
            --create-home ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" \
      > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    usermod -aG dialout,tty ${USERNAME} && \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

RUN groupadd -f -g ${VIDEO_GID} video && \
    usermod  -aG video ${USERNAME}

COPY start-vnc.sh /home/agilicious/start-vnc.sh
RUN chown agilicious:agilicious /home/agilicious/start-vnc.sh && \
    chmod +x /home/agilicious/start-vnc.sh

USER ${USERNAME}
WORKDIR ${HOME}

# clone + install your ZSH setup
RUN git clone https://github.com/AndruGomes13/zsh-quick-boot.git ~/zsh-quick-boot && \
    ~/zsh-quick-boot/install.sh

COPY --chown=${USERNAME}:${USERNAME} dotfiles/.zshrc /tmp/my_zshrc
RUN cat /tmp/my_zshrc >> ~/.zshrc && rm /tmp/my_zshrc


# set up your catkin workspace
RUN /bin/bash -lc "\
      source /opt/ros/${ROS_DISTRO}/setup.bash && \
      mkdir -p catkin_ws/src && \
      cd catkin_ws && \
      catkin config --init --mkdirs \
                    --extend /opt/ros/${ROS_DISTRO} \
                    --merge-devel \
                    --cmake-args -DCMAKE_BUILD_TYPE=Release"


ENTRYPOINT ["/home/agilicious/start-vnc.sh"]

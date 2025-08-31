# <<< Auto added .zhsrc <<<


set_ros_ip() {
  export ROS_IP=$(hostname -I | awk '{print $1}')
  echo "ROS_IP set to $ROS_IP"
}

WORKDIR="$HOME/catkin_ws"
ROS_SETUP="source $HOME/catkin_ws/devel/setup.sh"
INIT_SETUP="$ROS_SETUP && cd $WORKDIR  && set_ros_ip && clear"
eval $INIT_SETUP

# >>> Auto added .zshrc >>>
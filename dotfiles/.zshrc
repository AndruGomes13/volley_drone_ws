# <<< Auto added .zhsrc <<<


set_ros_ip() {
  export ROS_IP=$(hostname -I | awk '{print $1}')
  echo "ROS_IP set to $ROS_IP"
}

# >>> Auto added .zshrc >>>
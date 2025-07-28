cmake_minimum_required(VERSION 3.0.2)
project(util_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  agiros_msgs
)

catkin_package()

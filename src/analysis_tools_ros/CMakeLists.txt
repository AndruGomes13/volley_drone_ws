cmake_minimum_required(VERSION 3.0.2)
project(analysis_tools_ros)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()


install(
  DIRECTORY src/analysis_tools
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
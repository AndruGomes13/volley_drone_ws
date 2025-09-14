cmake_minimum_required(VERSION 3.0.2)
project(traj_policy_pkg)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()


install(
  DIRECTORY src/traj_policy_pkg
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
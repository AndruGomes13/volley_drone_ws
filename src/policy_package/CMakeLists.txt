cmake_minimum_required(VERSION 3.0.2)
project(policy_package)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()


install(
  DIRECTORY src/policy_package
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
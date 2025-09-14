cmake_minimum_required(VERSION 3.0.2)
project(drone_playground_py)

find_package(catkin REQUIRED)
catkin_package()

# Path to your Python project (the submodule with pyproject.toml)
set(PYPROJ_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../drone_2d_playground)

# Make sure we use the same interpreter catkin found
execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip --version
                RESULT_VARIABLE _pip_ok OUTPUT_QUIET ERROR_QUIET)
if(NOT _pip_ok EQUAL 0)
  message(FATAL_ERROR "pip not found for ${PYTHON_EXECUTABLE}.")
endif()

# Build the pip command
set(PIP_SPEC "${PYPROJ_DIR}")

# Editable install so the code updates without re-installing wheels
add_custom_target(
  install_drone_2d_playground ALL
  COMMAND ${PYTHON_EXECUTABLE} -m pip install -e ${PIP_SPEC}
  WORKING_DIRECTORY ${PYPROJ_DIR}
  COMMENT "pip installing ${PIP_SPEC}"
)

# Nothing to actually 'install' via CMake; Python takes care of it.
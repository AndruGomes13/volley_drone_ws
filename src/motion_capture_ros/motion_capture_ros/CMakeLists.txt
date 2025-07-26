cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0048 NEW)
project(motion_capture_ros VERSION 0.0.1 LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Configure Libmotioncapture
set(LIBMOTIONCAPTURE_ENABLE_VICON ON CACHE BOOL "" FORCE)
set(LIBMOTIONCAPTURE_ENABLE_QUALISYS OFF CACHE BOOL "" FORCE)
set(LIBMOTIONCAPTURE_ENABLE_OPTITRACK ON CACHE BOOL "" FORCE)
set(LIBMOTIONCAPTURE_ENABLE_OPTITRACK_CLOSED_SOURCE OFF CACHE BOOL "" FORCE)
set(LIBMOTIONCAPTURE_ENABLE_NOKOV OFF CACHE BOOL "" FORCE)
set(LIBMOTIONCAPTURE_ENABLE_VRPN OFF CACHE BOOL "" FORCE)
set(LIBMOTIONCAPTURE_ENABLE_FZMOTION OFF CACHE BOOL "" FORCE)
set(LIBMOTIONCAPTURE_ENABLE_MOTIONANALYSIS OFF CACHE BOOL "" FORCE)
set(LIBMOTIONCAPTURE_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(LIBMOTIONCAPTURE_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)

# Add the external libmotioncapture
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/libmotioncapture
  ${CMAKE_BINARY_DIR}/external/libmotioncapture
)

# Import catkin_simple for ROS dependencies
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

# Find Eigen3 (system dependency)
find_package(Eigen3 REQUIRED)

# Declare executable
cs_add_executable(motion_capture_node src/motion_capture.cpp)

target_include_directories(motion_capture_node PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/libmotioncapture/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/libmotioncapture/deps/vicon-datastream-sdk/Vicon/CrossMarket/DataStream
)

target_link_libraries(motion_capture_node
  libmotioncapture
)

# Export Eigen3 dependency to downstream packages
cs_export(DEPENDENCIES Eigen3)

# Install targets and headers
cs_install()
cmake_minimum_required(VERSION 3.16)
project(glim_segmentation)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

option(BUILD_WITH_CV_BRIDGE "Build with cv_bridge which enables support for processing of camera images" ON)

set(TRT_PATH "/usr/local/TensorRT" CACHE PATH "Path to TensorRT")
set(BUILD_PYTHON ON CACHE BOOL "Build python bindings")
add_subdirectory(3rdparty/TensorRT-YOLO)

if(DEFINED ENV{ROS_VERSION})
  if($ENV{ROS_VERSION} EQUAL 2)
    find_package(ament_cmake REQUIRED)
  elseif($ENV{ROS_VERSION} EQUAL 1)
    # ROS1
    message(FATAL_ERROR "Ros1 not implemented")
    find_package(catkin REQUIRED)
  endif()
endif()

find_package(rclcpp REQUIRED)
find_package(glim REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(realsense2_camera_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

if(DEFINED GLIM_USE_OPENCV AND NOT GLIM_USE_OPENCV AND BUILD_WITH_CV_BRIDGE)
  message(WARNING "BUILD_WITH_CV_BRIDGE is disabled because glim is not built with OpenCV.")
  set(BUILD_WITH_CV_BRIDGE OFF CACHE BOOL "glim is not built with OpenCV" FORCE)
endif()


if(BUILD_WITH_CV_BRIDGE)
  add_definitions(-DBUILD_WITH_CV_BRIDGE)
endif()

set(OLD_DISTRO "humble;galactic;foxy")
if($ENV{ROS_DISTRO} IN_LIST OLD_DISTRO)
  add_definitions(-DCV_BRIDGE_INCLUDE_H)
else()
  add_definitions(-DCV_BRIDGE_INCLUDE_HPP)
endif()

add_library(glim_segmentation SHARED
  src/util/config_segmentation.cpp
  src/segment_module.cpp
  src/segment_module_ros2.cpp
  src/segment_module_create.cpp
 # src/semantic_type.cpp
)

target_include_directories(glim_segmentation PUBLIC
  include
  ${PCL_INCLUDE_DIRS} 
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/TensorRT-YOLO/modules
  ${glim_INCLUDE_DIRS}
)

target_link_libraries(glim_segmentation
  glim::glim
  trtyolo
)


set_target_properties(glim_segmentation PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

list(APPEND glim_segmentation_LIBRARIES glim_segmentation)

if(DEFINED ENV{ROS_VERSION})
  if($ENV{ROS_VERSION} EQUAL 1)
    set(CMAKE_INSTALL_INCLUDEDIR "include/glim")
    set(CMAKE_INSTALL_LIBDIR "lib")
    set(CMAKE_INSTALL_BINDIR "bin/glim")
    set(CMAKE_INSTALL_DATADIR "share/glim")
    set(CMAKE_CONFIG_INSTALL_DIR "share/glim")
  elseif($ENV{ROS_VERSION} EQUAL 2)
    set(CMAKE_INSTALL_INCLUDEDIR "include/glim")
    set(CMAKE_INSTALL_LIBDIR "lib")
    set(CMAKE_INSTALL_BINDIR "bin/glim")
    set(CMAKE_INSTALL_DATADIR "share/glim")
    set(CMAKE_CONFIG_INSTALL_DIR "share/glim")
  else()
    message(FATAL_ERROR "Unsupported ROS version")
  endif()
else()
  include(GNUInstallDirs)
  set(CMAKE_CONFIG_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/glim"
    CACHE PATH "Install directory for CMake config files"
  )
endif()

install(TARGETS ${glim_segmentation_LIBRARIES}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

#################
## ROS-related ##
#################
if(DEFINED ENV{ROS_VERSION})
  if($ENV{ROS_VERSION} EQUAL 2)
    # ROS2
    install(DIRECTORY config DESTINATION share/glim_segmentation)
     ament_export_include_directories(include ${OpenCV_INCLUDE_DIRS})
    ament_target_dependencies(glim_segmentation 
      rclcpp
      glim
      pcl_conversions
      realsense2_camera_msgs
      cv_bridge
    )
    ament_export_libraries(${glim_segmentation_LIBRARIES})
    ament_package()
  elseif($ENV{ROS_VERSION} EQUAL 1)
    # ROS1
    message(FATAL_ERROR "Ros1 not implemented")
    catkin_package(
      LIBRARIES ${glim_segmentation_LIBRARIES}
    )
  endif()
endif()

cmake_minimum_required(VERSION 2.8.3)
project(soft_arm)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg geometry_msgs)

## Declare ROS messages and services
#add_message_files(FILES Num.msg)
add_service_files(FILES OneSeg.srv)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs geometry_msgs)

## Declare a catkin package
catkin_package(
INCLUDE_DIRS include
LIBRARIES PoseSubscriber
)

## Build talker and listener
#include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${soft_arm_INCLUDE_DIRS}
)

add_library(
soft_arm src/PoseSubscriber.cpp
)

add_executable(read_mocap src/read_mocap.cpp)
target_link_libraries(read_mocap soft_arm ${catkin_LIBRARIES})

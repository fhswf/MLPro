cmake_minimum_required(VERSION 3.5)
project(industrial_robot_status_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(catkin REQUIRED
  COMPONENTS
    controller_interface
    hardware_interface
    industrial_msgs
    industrial_robot_status_interface
    pluginlib
    realtime_tools
)

# this is empty as this pkg provides a plugin. Users are not expected
# to link the library against a binary directly.
catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/controller.cpp)
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(FILES plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

cmake_minimum_required(VERSION 2.8.3)
project(banana)

find_package(catkin REQUIRED COMPONENTS message_generation rospy)

# add_message_files(
#   FILES  # e.g. Floats.msg HeaderString.msg
# )


## Generate services in the 'srv' folder
# add_service_files(
#  FILES  # e.g. Floats.srv HeaderString.srv
#)

## Generate added messages and services with any dependencies
generate_messages()

catkin_package(
  CATKIN_DEPENDS message_runtime
)

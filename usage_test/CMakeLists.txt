cmake_minimum_required(VERSION 3.25)
project(cib_usage)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/get_cpm.cmake)
cpmaddpackage(NAME compile-time-init-build SOURCE_DIR
              "${CMAKE_CURRENT_LIST_DIR}/.." GIT_TAG HEAD)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE cib)

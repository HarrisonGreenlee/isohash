cmake_minimum_required(VERSION 3.14)
project(isohash)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

find_package(Armadillo REQUIRED)
message(STATUS "Armadillo libraries: ${ARMADILLO_LIBRARIES}")

pybind11_add_module(isohash bindings.cpp main.cpp)

target_include_directories(isohash PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(isohash PRIVATE ${ARMADILLO_LIBRARIES})

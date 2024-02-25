cmake_minimum_required(VERSION 3.22)

project(TimeSeriesDB
    VERSION 1.0
    DESCRIPTION "Time Series Indexing Database"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.56.0
)
set(FETCHCONTENT_QUIET OFF)
set(ABSL_ENABLE_INSTALL ON) 
FetchContent_MakeAvailable(gRPC)

#find_package(protobuf REQUIRED)
#find_package(gRPC REQUIRED)

find_package(glog  REQUIRED)

add_subdirectory(lib_indexing)
add_subdirectory(test)
add_subdirectory(server)

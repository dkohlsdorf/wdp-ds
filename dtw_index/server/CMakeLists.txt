add_library(
    TsIndexingServer
    SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/gen/indexing.grpc.pb.cc"	
    "${CMAKE_CURRENT_SOURCE_DIR}/gen/indexing.grpc.pb.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/gen/indexing.pb.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/gen/indexing.pb.cc"	
)

target_include_directories(TsIndexingServer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(TsIndexingServer glog::glog grpc++)

add_executable(server IndexingServer.cpp)
target_link_libraries(server PRIVATE TsIndexing TsIndexingServer glog::glog grpc++ protobuf)

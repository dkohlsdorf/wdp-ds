add_library(
    TsIndexing
    SHARED    
    "${CMAKE_CURRENT_SOURCE_DIR}/IndexingTypes.h"	
    "${CMAKE_CURRENT_SOURCE_DIR}/IndexingUtil.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/IndexingUtil.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/Serialization.h"	
    "${CMAKE_CURRENT_SOURCE_DIR}/Serialization.cpp"    
    "${CMAKE_CURRENT_SOURCE_DIR}/TimeSeriesIndex.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/TimeSeriesIndex.h"
)

target_include_directories(TsIndexing PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(TsIndexing glog::glog)


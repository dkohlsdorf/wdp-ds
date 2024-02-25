include(GoogleTest)

add_executable(benchmark PerformanceTesting.cpp)
target_link_libraries(benchmark PRIVATE TsIndexing GTest::gtest GTest::gtest_main glog::glog)

add_executable(indexing_util_test IndexingUtilTest.cpp)
target_link_libraries(indexing_util_test PRIVATE TsIndexing GTest::gtest GTest::gtest_main glog::glog)
gtest_discover_tests(indexing_util_test)

add_executable(serialization_test SerializationTest.cpp)
target_link_libraries(serialization_test PRIVATE TsIndexing GTest::gtest GTest::gtest_main glog::glog)
gtest_discover_tests(serialization_test)

cmake_minimum_required (VERSION 2.6)
project (hnsw_lib)

include_directories("${PROJECT_BINARY_DIR}")



set(SOURCE_EXE hnswlib/bench_distances.cpp)
add_subdirectory(benchmark)


add_executable(bench ${SOURCE_EXE})
SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=x86-64 -m64 -mavx -msse4 -mf16c -O3  -fpic -w  -Wall -fPIC -undefined" )
target_link_libraries(bench benchmark::benchmark)
#!/usr/bin/env bash
rm ./build/bench
clang++ hnswlib/bench_distances.cpp -lbenchmark -L/usr/local/lib/ -march=x86-64 -m64 -mavx -msse4 -mf16c -O3 -std=c++11 -Wall -fPIC -undefined dynamic_lookup -o build/bench
./build/bench --benchmark_repetitions=20 --benchmark_display_aggregates_only=true --benchmark_out_format=csv --benchmark_out=./bench-results.csv
#include <assert.h>
#include "../benchmark/include/benchmark/benchmark.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "hnswlib.h"

class L2BaseBench : public benchmark::Fixture {
    public:
    size_t dimension;
    size_t nb_embeddings = 100;

    hnswlib::L2Space* space;
    void* dist_func_param;
    hnswlib::DISTFUNC<float> dist_func;
    std::vector<std::vector<float>> scenario_input;
    size_t scenario_input_index;
    size_t scenario_input_size;

    const float RAND_MAX_FLOAT = (float)(RAND_MAX);

    void SetUp(const ::benchmark::State& state) {
        scenario_input.clear();
        scenario_input.reserve(nb_embeddings);
        for(int i = 0; i < nb_embeddings; i++) {
            std::vector<float> vector(dimension);
            for (int j = 0; j < dimension; j++) {
                vector[j] = (float)rand()/RAND_MAX_FLOAT;
            }
            scenario_input.push_back(vector);
        }
        space = new hnswlib::L2Space(dimension);
        dist_func_param = space->get_dist_func_param();
        dist_func = space->get_dist_func();

        scenario_input_size = scenario_input.size();
        scenario_input_index = 0;
    }

    void TearDown(const ::benchmark::State& state) {
        delete space;
    }

    float* get_vector() {
        float* vector_data = scenario_input[scenario_input_index].data();
        scenario_input_index++;
        if(scenario_input_index == scenario_input_size) {
            scenario_input_index = 0;
        }
        return vector_data;
    }

    float compute_distance() {
        auto vector1 = get_vector();
        auto vector2 = get_vector();
        return dist_func(vector1, vector2, dist_func_param);
    }
};

#define L2DimBench(dim)\
    class Dim##dim : public L2BaseBench {\
        public: Dim##dim() {\
            dimension = dim;\
        }\
    };\
    BENCHMARK_DEFINE_F(Dim##dim, Dist)(benchmark::State& st) { for (auto _ : st) compute_distance(); }\
    BENCHMARK_REGISTER_F(Dim##dim, Dist);

L2DimBench(3);
L2DimBench(4);
L2DimBench(7);
L2DimBench(8);
L2DimBench(9);
L2DimBench(15);
L2DimBench(16);
L2DimBench(100);
L2DimBench(101);
L2DimBench(128);
L2DimBench(129);

BENCHMARK_DEFINE_F(Dim3, Prep)(benchmark::State& st) {
    for (auto _ : st) {
        get_vector();
        get_vector();
    }
}

BENCHMARK_MAIN();
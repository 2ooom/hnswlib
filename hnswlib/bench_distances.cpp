#include <assert.h>
#include "../benchmark/include/benchmark/benchmark.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "hnswlib.h"

class BaseDistBench : public benchmark::Fixture {
    public:
    size_t dimension;
    size_t nb_embeddings = 100;
    bool baseline;

    hnswlib::SpaceInterface<float>* space;
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
        space = new hnswlib::L2Space(dimension, baseline);
        dist_func_param = &dimension;//space->get_dist_func_param();
        /*if(baseline) {
            dist_func = hnswlib::L2SqrSIMD4Ext;
        } else {
            dist_func = hnswlib::L2SqrSIMD8Ext;
        }
        */
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

#define L2Bench(dim)\
    class L2_test_##dim : public BaseDistBench {\
        public: L2_test_##dim() {\
            dimension = dim;\
            baseline = false;\
        }\
    };\
    \
    class L2_ref_##dim : public BaseDistBench {\
        public: L2_ref_##dim() {\
            dimension = dim;\
            baseline = true;\
        }\
    };\
    BENCHMARK_DEFINE_F(L2_ref_##dim, Dist)(benchmark::State& st) { for (auto _ : st) compute_distance(); }\
    BENCHMARK_DEFINE_F(L2_test_##dim, Dist)(benchmark::State& st) { for (auto _ : st) compute_distance(); }\
    BENCHMARK_REGISTER_F(L2_ref_##dim, Dist);\
    BENCHMARK_REGISTER_F(L2_test_##dim, Dist);


L2Bench(4);
L2Bench(8);
L2Bench(24);
/*
L2Bench(5);
L2Bench(6);
L2Bench(7);
L2Bench(8);
L2Bench(9);
L2Bench(10);
L2Bench(11);
L2Bench(12);
L2Bench(13);
L2Bench(14);
L2Bench(15);
L2Bench(16);
L2Bench(17);
L2Bench(18);
L2Bench(19);
L2Bench(20);
L2Bench(21);
L2Bench(22);
L2Bench(23);
L2Bench(24);
L2Bench(30);
L2Bench(32);
L2Bench(48);
L2Bench(64);
L2Bench(72);
L2Bench(80);
L2Bench(88);
L2Bench(95);
L2Bench(96);
L2Bench(100);
L2Bench(101);
L2Bench(112);
L2Bench(127);
L2Bench(128);
L2Bench(129);
L2Bench(256);
L2Bench(257);
L2Bench(300);
L2Bench(400);
L2Bench(512);
L2Bench(1024);
*/
/*
BENCHMARK_DEFINE_F(Dim3, Prep)(benchmark::State& st) {
    for (auto _ : st) {
        get_vector();
        get_vector();
    }
}*/

BENCHMARK_MAIN();
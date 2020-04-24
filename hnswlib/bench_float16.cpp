#include <assert.h>
#include "../benchmark/include/benchmark/benchmark.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "hnswlib.h"

class Float16Base : public benchmark::Fixture {
    public:
    size_t dimension;
    size_t nb_embeddings = 100;
    bool baseline;

    hnswlib::DECODEFUNC<float, uint16_t> encode_func;
    hnswlib::DECODEFUNC<uint16_t, float> decode_func;

    std::vector<std::vector<float>> encode_scenario_input;
    size_t encode_scenario_input_index;

    std::vector<std::vector<uint16_t>> decode_scenario_input;
    size_t decode_scenario_input_index;

    std::vector<float> result_decode;
    std::vector<uint16_t> result_encode;
    const float RAND_MAX_FLOAT = (float)(RAND_MAX);

    void SetUp(const ::benchmark::State& state) {
        encode_scenario_input.clear();
        encode_scenario_input.reserve(nb_embeddings);
        for(int i = 0; i < nb_embeddings; i++) {
            std::vector<float> vector(dimension);
            for (int j = 0; j < dimension; j++) {
                vector[j] = (float)rand()/RAND_MAX_FLOAT;
            }
            encode_scenario_input.push_back(vector);
        }
        encode_scenario_input_index = 0;
        result_encode.reserve(dimension);

        decode_scenario_input.clear();
        decode_scenario_input.reserve(nb_embeddings);
        for(int i = 0; i < nb_embeddings; i++) {
            std::vector<uint16_t> vector(dimension);
            for (int j = 0; j < dimension; j++) {
                vector[j] = (uint16_t)rand();
            }
            decode_scenario_input.push_back(vector);
        }
        decode_scenario_input_index = 0;
        result_decode.reserve(dimension);
        encode_func = hnswlib::get_fast_float16_encode_func(dimension, baseline);
        decode_func = hnswlib::get_fast_float16_decode_func(dimension, baseline);
    }

    float* get_vector_to_encode() {
        float* vector_data = encode_scenario_input[encode_scenario_input_index].data();
        encode_scenario_input_index++;
        if(encode_scenario_input_index == nb_embeddings) {
            encode_scenario_input_index = 0;
        }
        return vector_data;
    }

    uint16_t* get_vector_to_decode() {
        uint16_t* vector_data = decode_scenario_input[decode_scenario_input_index].data();
        decode_scenario_input_index++;
        if(decode_scenario_input_index == nb_embeddings) {
            decode_scenario_input_index = 0;
        }
        return vector_data;
    }

    void encode() {
        auto vector = get_vector_to_encode();
        encode_func(vector, result_encode.data(), dimension);
    }

    void decode() {
        auto vector = get_vector_to_decode();
        decode_func(vector, result_decode.data(), dimension);
    }
};


#define F16(dim)\
    class F16_ref_##dim : public Float16Base {\
        public: F16_ref_##dim() {\
            dimension = dim;\
            baseline = true;\
        }\
    };\
    BENCHMARK_DEFINE_F(F16_ref_##dim, encode)(benchmark::State& st) { for (auto _ : st) encode(); }\
    BENCHMARK_DEFINE_F(F16_ref_##dim, decode)(benchmark::State& st) { for (auto _ : st) decode(); }\
    \
    class F16_test_##dim : public Float16Base {\
        public: F16_test_##dim() {\
            dimension = dim;\
            baseline = false;\
        }\
    };\
    BENCHMARK_DEFINE_F(F16_test_##dim, encode)(benchmark::State& st) { for (auto _ : st) encode(); }\
    BENCHMARK_DEFINE_F(F16_test_##dim, decode)(benchmark::State& st) { for (auto _ : st) decode(); }\
    \
    BENCHMARK_REGISTER_F(F16_ref_##dim, encode);\
    BENCHMARK_REGISTER_F(F16_test_##dim, encode);\
    BENCHMARK_REGISTER_F(F16_ref_##dim, decode);\
    BENCHMARK_REGISTER_F(F16_test_##dim, decode);\


F16(3)
F16(7)
F16(15)
F16(23)
F16(31)
F16(63)
F16(75)
F16(95)
F16(100)
F16(101)
F16(103)
F16(127)


/*
BENCHMARK_DEFINE_F(Float16Base, prep_encode)(benchmark::State& st) {for (auto _ : st) { get_vector_to_encode();}}
BENCHMARK_REGISTER_F(Float16Base, prep_encode);

BENCHMARK_DEFINE_F(Float16Base, prep_decode)(benchmark::State& st) {for (auto _ : st) {get_vector_to_decode();}}
BENCHMARK_REGISTER_F(Float16Bse, prep_decode);*/

BENCHMARK_MAIN();
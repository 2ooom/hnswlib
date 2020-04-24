#pragma once
#include "hnswlib.h"

namespace hnswlib {

    template<typename src_type, typename dst_type, dst_type(*func)(src_type)>
    static void inline
    encode_decode(const src_type* src, dst_type* dst, const size_t qty) {
        std::transform(src, src + qty, dst, func);
    }

    static void
    encode_vector_float16(const float* src, uint16_t* dst, const size_t qty) {
        for (size_t i = 0; i < qty; i++) {
            *dst = encode_fp16(*src);
            src++;
            dst++;
        }
    }

    static void
    decode_float16_vector(const uint16_t* src, float* dst, const size_t qty) {
        for (size_t i = 0; i < qty; i++) {
            *dst = decode_fp16(*src);
            src++;
            dst++;
        }
    }


#if defined(USE_AVX)

    static void
    encode_vector_float16_SIMD8(const float* src, uint16_t* dst, const size_t qty) {
        const auto qty8 = qty >> 3;
        const float *end = src + (qty8 << 3);
        __m128i f16;
        __m256 f32;
        while (src < end) {
            f32 = _mm256_loadu_ps(src);
            src += 8;
            f16 = _mm256_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
            _mm_storeu_si128((__m128i*)dst, f16);
            dst +=8;
        }
    }

    static void
    decode_float16_vector_SIMD8(const uint16_t* src, float* dst, const size_t qty) {
        const auto qty8 = qty >> 3;
        const uint16_t *end = src + (qty8 << 3);
        __m128i f16;
        __m256 f32;
        while (src < end) {
            f16 = _mm_loadu_si128((__m128i*)src);
            src += 8;
            f32 = _mm256_cvtph_ps(f16);
            _mm256_storeu_ps(dst, f32);
            dst +=8;
        }
    }

    static void
    encode_vector_float16_SIMD8Residuals(const float* src, uint16_t* dst, const size_t qty) {
        const auto qty8 = qty >> 3 << 3;
        encode_vector_float16_SIMD8(src, dst, qty8);

        const auto qty_left = qty - qty8;
        encode_vector_float16(src + qty8, dst + qty8, qty_left);
    }

    static void
    decode_float16_vector_SIMD8Residuals(const uint16_t* src, float* dst, const size_t qty) {
        const auto qty8 = qty >> 3 << 3;
        decode_float16_vector_SIMD8(src, dst, qty8);

        const auto qty_left = qty - qty8;
        decode_float16_vector(src + qty8, dst + qty8, qty_left);
    }

    static void
    encode_vector_float16_SIMD8Residuals_new(const float* src, uint16_t* dst, const size_t qty) {
        const auto qty8 = qty >> 3 << 3;
        encode_vector_float16_SIMD8(src, dst, qty8);
        std::transform(src + qty8, src + qty, dst + qty8, encode_fp16);
    }

    static void
    decode_float16_vector_SIMD8Residuals_new(const uint16_t* src, float* dst, const size_t qty) {
        const auto qty8 = qty >> 3 << 3;
        decode_float16_vector_SIMD8(src, dst, qty8);
        std::transform(src + qty8, src + qty, dst + qty8, decode_fp16);
    }
#endif
#if defined(USE_SSE)

    static void
    encode_vector_float16_SIMD4(const float* src, uint16_t* dst, const size_t qty) {
        const auto qty4 = qty >> 2;
        const float *end = src + (qty4 << 2);
        __m128i f16;
        __m128 f32;
        while (src < end) {
            f32 = _mm_loadu_ps(src);
            src += 4;
            f16 = _mm_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
            _mm_storel_epi64((__m128i*)dst, f16);
            dst += 4;
        }
    }

    static void
    decode_float16_vector_SIMD4(const uint16_t* src, float* dst, const size_t qty) {
        const auto qty4 = qty >> 2;
        const uint16_t *end = src + (qty4 << 2);
        __m128i f16;
        __m128 f32;
        while (src < end) {
            f16 = _mm_loadu_si128((const __m128i*)src);
            src += 4;
            f32 = _mm_cvtph_ps(f16);
            _mm_storeu_ps(dst, f32);
            dst +=4;
        }
    }

    static void
    encode_vector_float16_SIMD4Residuals(const float* src, uint16_t* dst, const size_t qty) {
        const auto qty4 = qty >> 2 << 2;
        encode_vector_float16_SIMD4(src, dst, qty4);
        const auto qty_left = qty - qty4;
        encode_vector_float16(src + qty4, dst + qty4, qty_left);
    }

    static void
    decode_float16_vector_SIMD4Residuals(const uint16_t* src, float* dst, const size_t qty) {
        const auto qty4 = qty >> 2 << 2;
        decode_float16_vector_SIMD4(src, dst, qty4);
        const auto qty_left = qty - qty4;
        decode_float16_vector(src + qty4, dst + qty4, qty_left);
    }

    static void
    encode_vector_float16_SIMD4Residuals_new(const float* src, uint16_t* dst, const size_t qty) {
        const auto qty4 = qty >> 2 << 2;
        encode_vector_float16_SIMD4(src, dst, qty4);
        std::transform(src + qty4, src + qty, dst + qty4, encode_fp16);
    }

    static void
    decode_float16_vector_SIMD4Residuals_new(const uint16_t* src, float* dst, const size_t qty) {
        const auto qty4 = qty >> 2 << 2;
        decode_float16_vector_SIMD4(src, dst, qty4);
        std::transform(src + qty4, src + qty, dst + qty4, decode_fp16);
    }
#endif

    static
    DECODEFUNC<float, uint16_t> get_fast_float16_encode_func(size_t dim, bool baseline = true) {
        auto func = baseline? encode_vector_float16 : encode_decode<float, uint16_t, encode_fp16>;
    #if defined(USE_SSE)
        if (dim % 4 == 0) func = encode_vector_float16_SIMD4;
        else if (dim > 4) func = baseline?  encode_vector_float16_SIMD4Residuals : encode_vector_float16_SIMD4Residuals_new;
    #endif
    #if defined(USE_AVX)
        if (dim % 8 == 0) func = encode_vector_float16_SIMD8;
        else if (dim > 8) func = baseline? encode_vector_float16_SIMD8Residuals : encode_vector_float16_SIMD8Residuals_new;
    #endif
        return func;
    }

    static
    DECODEFUNC<uint16_t, float> get_fast_float16_decode_func(size_t dim, bool baseline = true) {
        auto func = baseline? decode_float16_vector : encode_decode<uint16_t, float, decode_fp16>;
    #if defined(USE_SSE)
        if (dim % 4 == 0) func = decode_float16_vector_SIMD4;
        else if (dim > 4) func = baseline?  decode_float16_vector_SIMD4Residuals : decode_float16_vector_SIMD4Residuals_new;
    #endif
    #if defined(USE_AVX)
        if (dim % 8 == 0) func = decode_float16_vector_SIMD8;
        else if (dim > 8) func = baseline?  decode_float16_vector_SIMD8Residuals : decode_float16_vector_SIMD8Residuals_new;
    #endif
        return func;
    }
}
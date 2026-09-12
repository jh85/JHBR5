// JHBR5 NNUE — AVX2 kernels. Semantics: see simd_scalar.h.

#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace jhbr5::simd::avx2 {

// A "tile" is 8 ymm registers of 16 int16 = 128 accumulator lanes. Weight rows
// are int8, so one tile consumes 128 bytes of each row per pass.
constexpr int kTileLanes = 128;
constexpr int kTileRegs = 8;

inline void CopyBias(int16_t* acc, const int16_t* bias, int n) {
  for (int i = 0; i < n; i += 16) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i),
                       _mm256_load_si256(reinterpret_cast<const __m256i*>(bias + i)));
  }
}

inline void UpdateRows(int16_t* acc, int n, const int8_t* base, size_t stride,
                       const int* adds, int n_adds, const int* subs,
                       int n_subs) {
  for (int t = 0; t < n; t += kTileLanes) {
    __m256i regs[kTileRegs];
    for (int k = 0; k < kTileRegs; ++k) {
      regs[k] = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + t + 16 * k));
    }
    for (int r = 0; r < n_adds; ++r) {
      const int8_t* row = base + static_cast<size_t>(adds[r]) * stride + t;
      for (int k = 0; k < kTileRegs; ++k) {
        const __m128i w8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + 16 * k));
        regs[k] = _mm256_add_epi16(regs[k], _mm256_cvtepi8_epi16(w8));
      }
    }
    for (int r = 0; r < n_subs; ++r) {
      const int8_t* row = base + static_cast<size_t>(subs[r]) * stride + t;
      for (int k = 0; k < kTileRegs; ++k) {
        const __m128i w8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + 16 * k));
        regs[k] = _mm256_sub_epi16(regs[k], _mm256_cvtepi8_epi16(w8));
      }
    }
    for (int k = 0; k < kTileRegs; ++k) {
      _mm256_store_si256(reinterpret_cast<__m256i*>(acc + t + 16 * k), regs[k]);
    }
  }
}

inline void PairwiseMul(const int16_t* acc, int n, int qa, int shift,
                        int16_t* out) {
  const int half = n / 2;
  const __m256i zero = _mm256_setzero_si256();
  const __m256i qav = _mm256_set1_epi16(static_cast<int16_t>(qa));
  for (int i = 0; i < half; i += 16) {
    __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
    __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + half + i));
    a = _mm256_min_epi16(_mm256_max_epi16(a, zero), qav);
    b = _mm256_min_epi16(_mm256_max_epi16(b, zero), qav);
    __m256i p = _mm256_mullo_epi16(a, b);  // <= qa*qa = 16384, fits int16
    if (shift) p = _mm256_srai_epi16(p, shift);
    _mm256_store_si256(reinterpret_cast<__m256i*>(out + i), p);
  }
}

inline void ScreluFull(const int16_t* acc, int n, int qa, int shift,
                       int16_t* out) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i qav = _mm256_set1_epi16(static_cast<int16_t>(qa));
  for (int i = 0; i < n; i += 16) {
    __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
    a = _mm256_min_epi16(_mm256_max_epi16(a, zero), qav);
    __m256i p = _mm256_mullo_epi16(a, a);  // <= qa*qa = 16384, fits int16
    if (shift) p = _mm256_srai_epi16(p, shift);
    _mm256_store_si256(reinterpret_cast<__m256i*>(out + i), p);
  }
}

inline int32_t HorizontalSum(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  lo = _mm_add_epi32(lo, hi);
  lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
  lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
  return _mm_cvtsi128_si32(lo);
}

inline int32_t DotI16I16(const int16_t* a, const int16_t* b, int n) {
  __m256i s0 = _mm256_setzero_si256();
  __m256i s1 = _mm256_setzero_si256();
  for (int i = 0; i < n; i += 32) {
    const __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i));
    const __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i + 16));
    const __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i + 16));
    s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(a0, b0));
    s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(a1, b1));
  }
  return HorizontalSum(_mm256_add_epi32(s0, s1));
}

inline int32_t DotI16I8(const int16_t* a, const int8_t* b, int n) {
  __m256i s0 = _mm256_setzero_si256();
  __m256i s1 = _mm256_setzero_si256();
  for (int i = 0; i < n; i += 32) {
    const __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i + 16));
    const __m256i b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
    const __m256i b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 16)));
    s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(a0, b0));
    s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(a1, b1));
  }
  return HorizontalSum(_mm256_add_epi32(s0, s1));
}

}  // namespace jhbr5::simd::avx2

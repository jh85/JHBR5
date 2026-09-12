// JHBR5 NNUE — SIMD abstraction.
//
// Every kernel exists in a scalar reference form (simd_scalar.h) and, when the
// build enables it, an ISA-specific form. The dispatch functions below pick the
// ISA form unless `force_scalar` is set (tests use this to compare the two on
// identical inputs).
//
// Conventions: `n` is the vector length in elements and must be a multiple of
// 32 (i16) so that no tail handling is needed; `stride` is the row length of a
// weight matrix in elements. All pointers must be 64-byte aligned.
//
// ISA selection: CMake option JHBR5_ISA=scalar|avx2|avx512 defines exactly one
// of JHBR5_SIMD_SCALAR / JHBR5_SIMD_AVX2 / JHBR5_SIMD_AVX512.

#pragma once

#include <cstddef>
#include <cstdint>

#include "nnue/simd_scalar.h"

#if defined(JHBR5_SIMD_AVX2)
#include "nnue/simd_avx2.h"
#define JHBR5_SIMD_NS avx2
#elif defined(JHBR5_SIMD_AVX512)
#include "nnue/simd_avx512.h"
#define JHBR5_SIMD_NS avx512
#else
#define JHBR5_SIMD_NS scalar
#endif

namespace jhbr5::simd {

// Set by tests to route every call through the scalar reference.
extern bool force_scalar;

inline const char* BackendName() {
#if defined(JHBR5_SIMD_AVX2)
  return "avx2";
#elif defined(JHBR5_SIMD_AVX512)
  return "avx512";
#else
  return "scalar";
#endif
}

// acc[0..n) = bias[0..n)
inline void CopyBias(int16_t* acc, const int16_t* bias, int n) {
  if (force_scalar) return scalar::CopyBias(acc, bias, n);
  JHBR5_SIMD_NS::CopyBias(acc, bias, n);
}

// acc += sum of rows base[adds[i]*stride ..] ; acc -= sum of rows base[subs[j]*stride ..]
inline void UpdateRows(int16_t* acc, int n, const int8_t* base, size_t stride,
                       const int* adds, int n_adds, const int* subs,
                       int n_subs) {
  if (force_scalar) {
    return scalar::UpdateRows(acc, n, base, stride, adds, n_adds, subs, n_subs);
  }
  JHBR5_SIMD_NS::UpdateRows(acc, n, base, stride, adds, n_adds, subs, n_subs);
}

// out[i] = (clamp(acc[i],0,qa) * clamp(acc[i+n/2],0,qa)) >> shift, i < n/2
inline void PairwiseMul(const int16_t* acc, int n, int qa, int shift,
                        int16_t* out) {
  if (force_scalar) return scalar::PairwiseMul(acc, n, qa, shift, out);
  JHBR5_SIMD_NS::PairwiseMul(acc, n, qa, shift, out);
}

// out[i] = clamp(acc[i],0,qa)^2 >> shift, i < n (full-width SCReLU, v2)
inline void ScreluFull(const int16_t* acc, int n, int qa, int shift,
                       int16_t* out) {
  if (force_scalar) return scalar::ScreluFull(acc, n, qa, shift, out);
  JHBR5_SIMD_NS::ScreluFull(acc, n, qa, shift, out);
}

// sum_i a[i]*b[i] as int32 (no overflow protection: callers size the scales)
inline int32_t DotI16I16(const int16_t* a, const int16_t* b, int n) {
  if (force_scalar) return scalar::DotI16I16(a, b, n);
  return JHBR5_SIMD_NS::DotI16I16(a, b, n);
}

inline int32_t DotI16I8(const int16_t* a, const int8_t* b, int n) {
  if (force_scalar) return scalar::DotI16I8(a, b, n);
  return JHBR5_SIMD_NS::DotI16I8(a, b, n);
}

inline void Prefetch(const void* p) {
#if defined(__GNUC__)
  __builtin_prefetch(p, 0, 1);
#else
  (void)p;
#endif
}

}  // namespace jhbr5::simd

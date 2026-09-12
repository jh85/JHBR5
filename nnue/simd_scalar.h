// JHBR5 NNUE — scalar reference kernels. These define the semantics; every
// ISA implementation must produce bit-identical results.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace jhbr5::simd::scalar {

inline void CopyBias(int16_t* acc, const int16_t* bias, int n) {
  for (int i = 0; i < n; ++i) acc[i] = bias[i];
}

inline void UpdateRows(int16_t* acc, int n, const int8_t* base, size_t stride,
                       const int* adds, int n_adds, const int* subs,
                       int n_subs) {
  for (int r = 0; r < n_adds; ++r) {
    const int8_t* row = base + static_cast<size_t>(adds[r]) * stride;
    for (int i = 0; i < n; ++i) {
      acc[i] = static_cast<int16_t>(acc[i] + row[i]);
    }
  }
  for (int r = 0; r < n_subs; ++r) {
    const int8_t* row = base + static_cast<size_t>(subs[r]) * stride;
    for (int i = 0; i < n; ++i) {
      acc[i] = static_cast<int16_t>(acc[i] - row[i]);
    }
  }
}

inline void PairwiseMul(const int16_t* acc, int n, int qa, int shift,
                        int16_t* out) {
  const int half = n / 2;
  for (int i = 0; i < half; ++i) {
    const int a = std::clamp<int>(acc[i], 0, qa);
    const int b = std::clamp<int>(acc[i + half], 0, qa);
    out[i] = static_cast<int16_t>((a * b) >> shift);
  }
}

inline void ScreluFull(const int16_t* acc, int n, int qa, int shift,
                       int16_t* out) {
  for (int i = 0; i < n; ++i) {
    const int a = std::clamp<int>(acc[i], 0, qa);
    out[i] = static_cast<int16_t>((a * a) >> shift);
  }
}

inline int32_t DotI16I16(const int16_t* a, const int16_t* b, int n) {
  int32_t sum = 0;
  for (int i = 0; i < n; ++i) sum += static_cast<int32_t>(a[i]) * b[i];
  return sum;
}

inline int32_t DotI16I8(const int16_t* a, const int8_t* b, int n) {
  int32_t sum = 0;
  for (int i = 0; i < n; ++i) sum += static_cast<int32_t>(a[i]) * b[i];
  return sum;
}

}  // namespace jhbr5::simd::scalar

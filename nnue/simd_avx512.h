// JHBR5 NNUE — AVX-512 kernels: STUB.
//
// TODO(avx512): implement with 512-bit tiles (kTileLanes = 256, 8 zmm
// registers), `_mm512_cvtepi8_epi16` from 256-bit loads for the row update,
// `_mm512_madd_epi16` (or VNNI `_mm512_dpwssd_epi32`) for the dot products,
// and `_mm512_reduce_add_epi32` for the horizontal sum. The development
// machine has no AVX-512, so these bodies are deliberately left unwritten
// rather than shipped untested. Selecting -DJHBR5_ISA=avx512 fails to compile
// on purpose until this file is completed and test_nnue_kernels passes on an
// AVX-512 host.

#pragma once

#include <cstddef>
#include <cstdint>

namespace jhbr5::simd::avx512 {

static_assert(sizeof(void*) == 0,
              "JHBR5_ISA=avx512 selected but nnue/simd_avx512.h is a stub; "
              "implement and validate it on an AVX-512 machine first");

inline void CopyBias(int16_t*, const int16_t*, int) {}
inline void UpdateRows(int16_t*, int, const int8_t*, size_t, const int*, int,
                       const int*, int) {}
inline void PairwiseMul(const int16_t*, int, int, int, int16_t*) {}
inline int32_t DotI16I16(const int16_t*, const int16_t*, int) { return 0; }
inline int32_t DotI16I8(const int16_t*, const int8_t*, int) { return 0; }

}  // namespace jhbr5::simd::avx512

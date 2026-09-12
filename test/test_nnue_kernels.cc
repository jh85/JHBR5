// Scalar reference vs ISA kernels: results must be bit-identical.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nnue/aligned.h"
#include "nnue/simd.h"

namespace {

struct Rng {
  uint64_t s;
  uint64_t Next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  int Int(int lo, int hi) { return lo + static_cast<int>(Next() % (hi - lo + 1)); }
};

int failures = 0;

void Check(bool ok, const char* what, int n, int seed) {
  if (!ok) {
    std::printf("FAIL %s n=%d seed=%d\n", what, n, seed);
    ++failures;
  }
}

void RunCase(int n, int seed) {
  using namespace jhbr5;
  Rng rng{static_cast<uint64_t>(seed) * 1000003ULL + n};
  const int rows = 64;
  nnue::AlignedBuffer<int8_t> base(static_cast<size_t>(rows) * n);
  nnue::AlignedBuffer<int16_t> bias(n), acc_s(n), acc_v(n), out_s(n / 2), out_v(n / 2), b16(n);
  nnue::AlignedBuffer<int16_t> full_s(n), full_v(n);
  for (size_t i = 0; i < base.size(); ++i) base[i] = static_cast<int8_t>(rng.Int(-128, 127));
  for (int i = 0; i < n; ++i) {
    bias[i] = static_cast<int16_t>(rng.Int(-2000, 2000));
    b16[i] = static_cast<int16_t>(rng.Int(-32768, 32767));
  }
  std::vector<int> adds(20), subs(17);
  for (int& a : adds) a = rng.Int(0, rows - 1);
  for (int& s : subs) s = rng.Int(0, rows - 1);

  simd::force_scalar = true;
  simd::CopyBias(acc_s.data(), bias.data(), n);
  simd::UpdateRows(acc_s.data(), n, base.data(), n, adds.data(), 20, subs.data(), 17);
  simd::force_scalar = false;
  simd::CopyBias(acc_v.data(), bias.data(), n);
  simd::UpdateRows(acc_v.data(), n, base.data(), n, adds.data(), 20, subs.data(), 17);
  Check(std::memcmp(acc_s.data(), acc_v.data(), n * 2) == 0, "UpdateRows", n, seed);

  for (int shift : {0, 2}) {
    simd::force_scalar = true;
    simd::PairwiseMul(acc_s.data(), n, 128, shift, out_s.data());
    simd::force_scalar = false;
    simd::PairwiseMul(acc_s.data(), n, 128, shift, out_v.data());
    Check(std::memcmp(out_s.data(), out_v.data(), n) == 0, "PairwiseMul", n, seed);

    simd::force_scalar = true;
    simd::ScreluFull(acc_s.data(), n, 128, shift, full_s.data());
    simd::force_scalar = false;
    simd::ScreluFull(acc_s.data(), n, 128, shift, full_v.data());
    Check(std::memcmp(full_s.data(), full_v.data(), n * 2) == 0, "ScreluFull", n, seed);
  }

  simd::force_scalar = true;
  const int32_t d1 = simd::DotI16I16(acc_s.data(), b16.data(), n);
  const int32_t d2 = simd::DotI16I8(acc_s.data(), base.data(), n);
  simd::force_scalar = false;
  Check(simd::DotI16I16(acc_s.data(), b16.data(), n) == d1, "DotI16I16", n, seed);
  Check(simd::DotI16I8(acc_s.data(), base.data(), n) == d2, "DotI16I8", n, seed);
}

}  // namespace

int main() {
  for (int n : {256, 512, 1024, 2048}) {
    for (int seed = 0; seed < 8; ++seed) RunCase(n, seed);
  }
  std::printf("test_nnue_kernels backend=%s: %s\n", jhbr5::simd::BackendName(),
              failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}

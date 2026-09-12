#include "nnue/random_net.h"

#include <cstring>
#include <vector>

#include "nnue/net_format.h"
#include "nnue/types.h"

namespace jhbr5::nnue {

namespace {

struct SplitMix64 {
  uint64_t s;
  uint64_t Next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  int Int(int lo, int hi) {  // inclusive
    return lo + static_cast<int>(Next() % static_cast<uint64_t>(hi - lo + 1));
  }
  float Uniform(float lo, float hi) {
    return lo + (hi - lo) * static_cast<float>(Next() >> 40) / static_cast<float>(1u << 24);
  }
};

template <typename T>
void Fill(std::vector<T>& v, SplitMix64& rng, int lo, int hi) {
  for (auto& x : v) x = static_cast<T>(rng.Int(lo, hi));
}

void FillF(std::vector<float>& v, SplitMix64& rng, float lo, float hi) {
  for (auto& x : v) x = rng.Uniform(lo, hi);
}

NetHeader BaseHeader(uint32_t kind) {
  NetHeader h{};
  h.kind = kind;
  h.feature_set_id = FeatureSetId();
  h.qa = kQA;
  h.q_pst = kQPst;
  return h;
}

}  // namespace

bool WriteRandomValueNet(const std::string& path, int l1, uint64_t seed,
                         std::string* err) {
  SplitMix64 rng{seed};
  const size_t L1 = static_cast<size_t>(l1);
  const size_t half3 = 3 * L1 / 2;
  std::vector<int8_t> a_w(static_cast<size_t>(kGroupAInputs) * L1);
  std::vector<int16_t> a_b(L1);
  std::vector<int8_t> b_w(static_cast<size_t>(kGroupBInputs) * L1);
  std::vector<int16_t> b_b(L1);
  std::vector<int16_t> l2_w(static_cast<size_t>(kValueL2) * half3);
  std::vector<int16_t> l2_b(kValueL2);
  std::vector<float> l3_w(static_cast<size_t>(kValueL3) * kValueL2), l3_b(kValueL3);
  std::vector<float> l4_w(static_cast<size_t>(3) * kValueL3), l4_b(3);
  std::vector<int16_t> pst(static_cast<size_t>(kGroupBInputs) * 3);
  Fill(a_w, rng, -6, 6);
  Fill(a_b, rng, 0, 64);
  Fill(b_w, rng, -6, 6);
  Fill(b_b, rng, 0, 64);
  Fill(l2_w, rng, -200, 200);
  Fill(l2_b, rng, -100, 100);
  FillF(l3_w, rng, -0.3f, 0.3f);
  FillF(l3_b, rng, -0.3f, 0.3f);
  FillF(l4_w, rng, -0.3f, 0.3f);
  FillF(l4_b, rng, -0.3f, 0.3f);
  Fill(pst, rng, -8, 8);

  NetWriter w;
  w.AddTensor("a_w", DType::kI8, {kGroupAInputs, L1}, a_w.data());
  w.AddTensor("a_b", DType::kI16, {L1}, a_b.data());
  w.AddTensor("b_w", DType::kI8, {kGroupBInputs, L1}, b_w.data());
  w.AddTensor("b_b", DType::kI16, {L1}, b_b.data());
  w.AddTensor("l2_w", DType::kI16, {kValueL2, half3}, l2_w.data());
  w.AddTensor("l2_b", DType::kI16, {kValueL2}, l2_b.data());
  w.AddTensor("l3_w", DType::kF32, {kValueL3, kValueL2}, l3_w.data());
  w.AddTensor("l3_b", DType::kF32, {kValueL3}, l3_b.data());
  w.AddTensor("l4_w", DType::kF32, {3, kValueL3}, l4_w.data());
  w.AddTensor("l4_b", DType::kF32, {3}, l4_b.data());
  w.AddTensor("pst", DType::kI16, {kGroupBInputs, 3}, pst.data());

  NetHeader h = BaseHeader(kNetKindValue);
  h.l1[0] = h.l1[1] = h.l1[2] = static_cast<uint32_t>(l1);
  h.l2 = kValueL2;
  h.l3 = kValueL3;
  h.qb = 256;
  return w.Write(path, h, err);
}

bool WriteRandomPolicyNet(const std::string& path, int l1, bool see_doubling,
                          uint64_t seed, std::string* err) {
  SplitMix64 rng{seed ^ 0x5eedULL};
  const size_t L1 = static_cast<size_t>(l1);
  const size_t rows = see_doubling ? kNumBucketsSee : kNumBuckets;
  std::vector<int8_t> l1_w(static_cast<size_t>(kPolicyInputs) * L1);
  std::vector<int16_t> l1_b(L1);
  std::vector<int8_t> out_w(rows * (L1 / 2));
  std::vector<int16_t> out_b(rows);
  Fill(l1_w, rng, -6, 6);
  Fill(l1_b, rng, 0, 64);
  Fill(out_w, rng, -10, 10);
  Fill(out_b, rng, -100, 100);

  NetWriter w;
  w.AddTensor("l1_w", DType::kI8, {kPolicyInputs, L1}, l1_w.data());
  w.AddTensor("l1_b", DType::kI16, {L1}, l1_b.data());
  w.AddTensor("out_w", DType::kI8, {rows, L1 / 2}, out_w.data());
  w.AddTensor("out_b", DType::kI16, {rows}, out_b.data());

  NetHeader h = BaseHeader(kNetKindPolicy);
  h.bucket_table_id = BucketTableId(see_doubling);
  h.l1[0] = static_cast<uint32_t>(l1);
  h.qb = kPolicyQB;
  return w.Write(path, h, err);
}

bool WriteRandomValueNetV2(const std::string& path, int l1, uint64_t seed,
                           std::string* err) {
  SplitMix64 rng{seed};
  const size_t L1 = static_cast<size_t>(l1);
  const size_t full3 = 3 * L1;
  std::vector<int8_t> a_w(static_cast<size_t>(kGroupA2Inputs) * L1);
  std::vector<int16_t> a_b(L1);
  std::vector<int8_t> b_w(static_cast<size_t>(kGroupBInputs) * L1);
  std::vector<int16_t> b_b(L1);
  std::vector<int16_t> l2_w(static_cast<size_t>(kValue2L2) * full3);
  std::vector<int16_t> l2_b(kValue2L2);
  std::vector<float> l3_w(static_cast<size_t>(kValueL3) * kValue2L2), l3_b(kValueL3);
  std::vector<float> l4_w(static_cast<size_t>(kPhaseBuckets) * 3 * kValueL3);
  std::vector<float> l4_b(static_cast<size_t>(kPhaseBuckets) * 3);
  std::vector<int16_t> pst(static_cast<size_t>(kGroupBInputs) * 3);
  Fill(a_w, rng, -6, 6);
  Fill(a_b, rng, 0, 64);
  Fill(b_w, rng, -6, 6);
  Fill(b_b, rng, 0, 64);
  // Twice the v1 L2 input width -> half the weight range for the same dot
  // headroom (int32 worst case 3*l1*16384*100 == v1's 3*l1/2*16384*200).
  Fill(l2_w, rng, -100, 100);
  Fill(l2_b, rng, -100, 100);
  FillF(l3_w, rng, -0.3f, 0.3f);
  FillF(l3_b, rng, -0.3f, 0.3f);
  FillF(l4_w, rng, -0.3f, 0.3f);
  FillF(l4_b, rng, -0.3f, 0.3f);
  Fill(pst, rng, -8, 8);

  NetWriter w;
  w.AddTensor("a_w", DType::kI8, {kGroupA2Inputs, L1}, a_w.data());
  w.AddTensor("a_b", DType::kI16, {L1}, a_b.data());
  w.AddTensor("b_w", DType::kI8, {kGroupBInputs, L1}, b_w.data());
  w.AddTensor("b_b", DType::kI16, {L1}, b_b.data());
  w.AddTensor("l2_w", DType::kI16, {kValue2L2, full3}, l2_w.data());
  w.AddTensor("l2_b", DType::kI16, {kValue2L2}, l2_b.data());
  w.AddTensor("l3_w", DType::kF32, {kValueL3, kValue2L2}, l3_w.data());
  w.AddTensor("l3_b", DType::kF32, {kValueL3}, l3_b.data());
  w.AddTensor("l4_w", DType::kF32, {kPhaseBuckets, 3, kValueL3}, l4_w.data());
  w.AddTensor("l4_b", DType::kF32, {kPhaseBuckets, 3}, l4_b.data());
  w.AddTensor("pst", DType::kI16, {kGroupBInputs, 3}, pst.data());

  NetHeader h = BaseHeader(kNetKindValue);
  h.version = kNetVersion2;
  h.n_phase = kPhaseBuckets;
  h.feature_set_id = FeatureSetIdV2();
  h.l1[0] = h.l1[1] = h.l1[2] = static_cast<uint32_t>(l1);
  h.l2 = kValue2L2;
  h.l3 = kValueL3;
  h.qb = 256;
  return w.Write(path, h, err);
}

bool WriteRandomPolicyNetV2(const std::string& path, int l1, bool see_doubling,
                            uint64_t seed, std::string* err) {
  SplitMix64 rng{seed ^ 0x5eedULL};
  const size_t L1 = static_cast<size_t>(l1);
  const size_t rows = see_doubling ? kNumBucketsSee : kNumBuckets;
  std::vector<int8_t> l1_w(static_cast<size_t>(kPolicy2Inputs) * L1);
  std::vector<int16_t> l1_b(L1);
  std::vector<int8_t> out_w(rows * L1);
  std::vector<int16_t> out_b(rows);
  Fill(l1_w, rng, -6, 6);
  Fill(l1_b, rng, 0, 64);
  Fill(out_w, rng, -10, 10);
  Fill(out_b, rng, -100, 100);

  NetWriter w;
  w.AddTensor("l1_w", DType::kI8, {kPolicy2Inputs, L1}, l1_w.data());
  w.AddTensor("l1_b", DType::kI16, {L1}, l1_b.data());
  w.AddTensor("out_w", DType::kI8, {rows, L1}, out_w.data());
  w.AddTensor("out_b", DType::kI16, {rows}, out_b.data());

  NetHeader h = BaseHeader(kNetKindPolicy);
  h.version = kNetVersion2;
  h.feature_set_id = FeatureSetIdV2();
  h.bucket_table_id = BucketTableId(see_doubling);
  h.l1[0] = static_cast<uint32_t>(l1);
  h.qb = kPolicyQB;
  return w.Write(path, h, err);
}

}  // namespace jhbr5::nnue

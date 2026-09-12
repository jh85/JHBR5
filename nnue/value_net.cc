#include "nnue/value_net.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "nnue/simd.h"

namespace jhbr5::nnue {

bool ValueNet::Load(const std::string& path, std::string* err) {
  if (!file_.Load(path, err)) return false;
  const NetHeader& h = file_.header();
  if (h.kind != kNetKindValue) {
    *err = path + ": not a value network";
    return false;
  }
  version_ = static_cast<int>(h.version);
  return version_ == 2 ? LoadV2(path, err) : LoadV1(path, err);
}

bool ValueNet::LoadV1(const std::string& path, std::string* err) {
  const NetHeader& h = file_.header();
  if (h.feature_set_id != FeatureSetId()) {
    *err = path + ": feature set id mismatch (net was trained for a different mapping)";
    return false;
  }
  l1_ = static_cast<int>(h.l1[0]);
  qb_ = static_cast<int>(h.qb);
  if (l1_ <= 0 || l1_ % 256 != 0 || h.l1[1] != h.l1[0] || h.l1[2] != h.l1[0] ||
      h.l2 != kValueL2 || h.l3 != kValueL3 || h.qa != kQA || h.q_pst != kQPst ||
      qb_ <= 0) {
    *err = path + ": unsupported value-net dimensions";
    return false;
  }
  const auto L1 = static_cast<uint64_t>(l1_);
  const uint64_t half3 = 3 * L1 / 2;
  a_w = file_.TensorAs<int8_t>("a_w", DType::kI8, {kGroupAInputs, L1}, err);
  a_b = file_.TensorAs<int16_t>("a_b", DType::kI16, {L1}, err);
  b_w = file_.TensorAs<int8_t>("b_w", DType::kI8, {kGroupBInputs, L1}, err);
  b_b = file_.TensorAs<int16_t>("b_b", DType::kI16, {L1}, err);
  l2_w = file_.TensorAs<int16_t>("l2_w", DType::kI16, {kValueL2, half3}, err);
  l2_b = file_.TensorAs<int16_t>("l2_b", DType::kI16, {kValueL2}, err);
  l3_w = file_.TensorAs<float>("l3_w", DType::kF32, {kValueL3, kValueL2}, err);
  l3_b = file_.TensorAs<float>("l3_b", DType::kF32, {kValueL3}, err);
  l4_w = file_.TensorAs<float>("l4_w", DType::kF32, {3, kValueL3}, err);
  l4_b = file_.TensorAs<float>("l4_b", DType::kF32, {3}, err);
  pst = file_.TensorAs<int16_t>("pst", DType::kI16, {kGroupBInputs, 3}, err);
  return a_w && a_b && b_w && b_b && l2_w && l2_b && l3_w && l3_b && l4_w &&
         l4_b && pst;
}

bool ValueNet::LoadV2(const std::string& path, std::string* err) {
  const NetHeader& h = file_.header();
  if (h.feature_set_id != FeatureSetIdV2()) {
    *err = path + ": feature set id mismatch (net was trained for a different mapping)";
    return false;
  }
  l1_ = static_cast<int>(h.l1[0]);
  qb_ = static_cast<int>(h.qb);
  if (l1_ <= 0 || l1_ % 256 != 0 || h.l1[1] != h.l1[0] || h.l1[2] != h.l1[0] ||
      h.l2 != kValue2L2 || h.l3 != kValueL3 || h.qa != kQA || h.q_pst != kQPst ||
      h.n_phase != kPhaseBuckets || qb_ <= 0) {
    *err = path + ": unsupported value-net dimensions";
    return false;
  }
  const auto L1 = static_cast<uint64_t>(l1_);
  const uint64_t full3 = 3 * L1;
  a_w = file_.TensorAs<int8_t>("a_w", DType::kI8, {kGroupA2Inputs, L1}, err);
  a_b = file_.TensorAs<int16_t>("a_b", DType::kI16, {L1}, err);
  b_w = file_.TensorAs<int8_t>("b_w", DType::kI8, {kGroupBInputs, L1}, err);
  b_b = file_.TensorAs<int16_t>("b_b", DType::kI16, {L1}, err);
  l2_w = file_.TensorAs<int16_t>("l2_w", DType::kI16, {kValue2L2, full3}, err);
  l2_b = file_.TensorAs<int16_t>("l2_b", DType::kI16, {kValue2L2}, err);
  l3_w = file_.TensorAs<float>("l3_w", DType::kF32, {kValueL3, kValue2L2}, err);
  l3_b = file_.TensorAs<float>("l3_b", DType::kF32, {kValueL3}, err);
  l4_w = file_.TensorAs<float>("l4_w", DType::kF32,
                               {kPhaseBuckets, 3, kValueL3}, err);
  l4_b = file_.TensorAs<float>("l4_b", DType::kF32, {kPhaseBuckets, 3}, err);
  pst = file_.TensorAs<int16_t>("pst", DType::kI16, {kGroupBInputs, 3}, err);
  return a_w && a_b && b_w && b_b && l2_w && l2_b && l3_w && l3_b && l4_w &&
         l4_b && pst;
}

ValueScratch::ValueScratch(const ValueNet& net)
    : net_(&net),
      l1_(net.l1()),
      v2_(net.is_v2()),
      cache_acc_(static_cast<size_t>(2) * (v2_ ? kKingBuckets : 81) * net.l1()),
      acc_b_(net.l1()),
      act_(v2_ ? static_cast<size_t>(3) * net.l1()
               : static_cast<size_t>(3) * net.l1() / 2) {
  Reset();
}

void ValueScratch::Reset() {
  for (auto& frame : state_) {
    for (auto& st : frame) st = FrameState{};
  }
  rows_a_ = rows_b_ = evals_ = 0;
}

const int16_t* ValueScratch::RefreshFrame(const ShogiBoard& board, Color p) {
  const int k = KingFrameSq(board, p);
  int16_t* acc = cache_acc_.data() + (static_cast<size_t>(p) * 81 + k) * l1_;
  FrameState* st = &state_[p][k];
  if (!st->initialized) {
    simd::CopyBias(acc, net_->a_b, l1_);
    std::memset(st->pieces, 0, sizeof(st->pieces));
    std::memset(st->hand, 0, sizeof(st->hand));
  }
  GroupADiff(board, p, st, &adds_, &subs_);
  const size_t stride = static_cast<size_t>(l1_);
  for (int i = 0; i < adds_.n; ++i) simd::Prefetch(net_->a_w + adds_.idx[i] * stride);
  for (int i = 0; i < subs_.n; ++i) simd::Prefetch(net_->a_w + subs_.idx[i] * stride);
  simd::UpdateRows(acc, l1_, net_->a_w, stride, adds_.idx, adds_.n, subs_.idx,
                   subs_.n);
  rows_a_ += static_cast<uint64_t>(adds_.n + subs_.n);
  return acc;
}

const int16_t* ValueScratch::RefreshFrameV2(const ShogiBoard& board, Color p) {
  const int k = KingBucket(KingFrameSq(board, p));
  int16_t* acc = cache_acc_.data() + (static_cast<size_t>(p) * kKingBuckets + k) * l1_;
  FrameState* st = &state_[p][k];
  if (!st->initialized) {
    simd::CopyBias(acc, net_->a_b, l1_);
    std::memset(st->pieces, 0, sizeof(st->pieces));
    std::memset(st->hand, 0, sizeof(st->hand));
  }
  GroupA2Diff(board, p, st, &adds_, &subs_);
  const size_t stride = static_cast<size_t>(l1_);
  for (int i = 0; i < adds_.n; ++i) simd::Prefetch(net_->a_w + adds_.idx[i] * stride);
  for (int i = 0; i < subs_.n; ++i) simd::Prefetch(net_->a_w + subs_.idx[i] * stride);
  simd::UpdateRows(acc, l1_, net_->a_w, stride, adds_.idx, adds_.n, subs_.idx,
                   subs_.n);
  rows_a_ += static_cast<uint64_t>(adds_.n + subs_.n);
  return acc;
}

namespace {

inline float SCReLU(float x) {
  x = std::clamp(x, 0.0f, 1.0f);
  return x * x;
}

}  // namespace

Wdl ValueScratch::Evaluate(const ShogiBoard& board) {
  if (v2_) return EvaluateV2(board);
  ++evals_;
  const Color stm = board.side_to_move();
  const int16_t* acc_us = RefreshFrame(board, stm);
  const int16_t* acc_them = RefreshFrame(board, ~stm);

  GroupBFeatures(board, &feats_b_);
  const size_t stride = static_cast<size_t>(l1_);
  for (int i = 0; i < feats_b_.n; ++i) simd::Prefetch(net_->b_w + feats_b_.idx[i] * stride);
  int16_t* acc_b = acc_b_.data();
  simd::CopyBias(acc_b, net_->b_b, l1_);
  simd::UpdateRows(acc_b, l1_, net_->b_w, stride, feats_b_.idx, feats_b_.n, nullptr, 0);
  rows_b_ += static_cast<uint64_t>(feats_b_.n);

  const int half = l1_ / 2;
  int16_t* act = act_.data();
  simd::PairwiseMul(acc_us, l1_, kQA, 0, act);
  simd::PairwiseMul(acc_them, l1_, kQA, 0, act + half);
  simd::PairwiseMul(acc_b, l1_, kQA, 0, act + 2 * half);

  const int n2 = 3 * half;
  const float inv_qa2 = 1.0f / static_cast<float>(kQA * kQA);
  const float inv_qb = 1.0f / static_cast<float>(net_->qb());
  float x[kValueL2];
  for (int j = 0; j < kValueL2; ++j) {
    const int32_t s = simd::DotI16I16(net_->l2_w + static_cast<size_t>(j) * n2, act, n2);
    x[j] = (static_cast<float>(s) * inv_qa2 + static_cast<float>(net_->l2_b[j])) * inv_qb;
  }

  float y[kValueL3];
  for (int j = 0; j < kValueL3; ++j) {
    float s = net_->l3_b[j];
    const float* w = net_->l3_w + static_cast<size_t>(j) * kValueL2;
    for (int i = 0; i < kValueL2; ++i) s += w[i] * SCReLU(x[i]);
    y[j] = s;
  }

  float z[3];
  for (int k = 0; k < 3; ++k) {
    float s = net_->l4_b[k];
    const float* w = net_->l4_w + static_cast<size_t>(k) * kValueL3;
    for (int j = 0; j < kValueL3; ++j) s += w[j] * SCReLU(y[j]);
    z[k] = s;
  }

  int32_t pst_sum[3] = {0, 0, 0};
  for (int i = 0; i < feats_b_.n; ++i) {
    const int16_t* p = net_->pst + static_cast<size_t>(feats_b_.idx[i]) * 3;
    pst_sum[0] += p[0];
    pst_sum[1] += p[1];
    pst_sum[2] += p[2];
  }
  const float inv_qpst = 1.0f / static_cast<float>(kQPst);
  for (int k = 0; k < 3; ++k) z[k] += static_cast<float>(pst_sum[k]) * inv_qpst;

  const float m = std::max(z[0], std::max(z[1], z[2]));
  const float e0 = std::exp(z[0] - m);
  const float e1 = std::exp(z[1] - m);
  const float e2 = std::exp(z[2] - m);
  const float inv = 1.0f / (e0 + e1 + e2);
  Wdl out;
  out.win = e0 * inv;
  out.draw = e1 * inv;
  out.loss = e2 * inv;
  return out;
}

// v2 evaluation (docs/NNUE_V2_DESIGN.md §2): king-bucketed finny cache,
// full-width SCReLU, residual L3, phase-conditioned L4.
Wdl ValueScratch::EvaluateV2(const ShogiBoard& board) {
  ++evals_;
  const Color stm = board.side_to_move();
  const int16_t* acc_us = RefreshFrameV2(board, stm);
  const int16_t* acc_them = RefreshFrameV2(board, ~stm);

  GroupBFeatures(board, &feats_b_);
  const size_t stride = static_cast<size_t>(l1_);
  for (int i = 0; i < feats_b_.n; ++i) simd::Prefetch(net_->b_w + feats_b_.idx[i] * stride);
  int16_t* acc_b = acc_b_.data();
  simd::CopyBias(acc_b, net_->b_b, l1_);
  simd::UpdateRows(acc_b, l1_, net_->b_w, stride, feats_b_.idx, feats_b_.n, nullptr, 0);
  rows_b_ += static_cast<uint64_t>(feats_b_.n);

  int16_t* act = act_.data();
  simd::ScreluFull(acc_us, l1_, kQA, 0, act);
  simd::ScreluFull(acc_them, l1_, kQA, 0, act + l1_);
  simd::ScreluFull(acc_b, l1_, kQA, 0, act + 2 * l1_);

  const int n2 = 3 * l1_;
  const float inv_qa2 = 1.0f / static_cast<float>(kQA * kQA);
  const float inv_qb = 1.0f / static_cast<float>(net_->qb());
  float x[kValue2L2];
  for (int j = 0; j < kValue2L2; ++j) {
    const int32_t s = simd::DotI16I16(net_->l2_w + static_cast<size_t>(j) * n2, act, n2);
    x[j] = (static_cast<float>(s) * inv_qa2 + static_cast<float>(net_->l2_b[j])) * inv_qb;
  }

  float y1[kValue2L2];
  for (int j = 0; j < kValue2L2; ++j) y1[j] = SCReLU(x[j]);

  // Residual L3: h = y1 + screlu(l3(y1)).
  float h[kValue2L2];
  for (int j = 0; j < kValue2L2; ++j) {
    float s = net_->l3_b[j];
    const float* w = net_->l3_w + static_cast<size_t>(j) * kValue2L2;
    for (int i = 0; i < kValue2L2; ++i) s += w[i] * y1[i];
    h[j] = y1[j] + SCReLU(s);
  }

  const int ph = PhaseBucket(board);
  float z[3];
  for (int k = 0; k < 3; ++k) {
    float s = net_->l4_b[ph * 3 + k];
    const float* w = net_->l4_w + static_cast<size_t>(ph * 3 + k) * kValueL3;
    for (int j = 0; j < kValueL3; ++j) s += w[j] * h[j];
    z[k] = s;
  }

  int32_t pst_sum[3] = {0, 0, 0};
  for (int i = 0; i < feats_b_.n; ++i) {
    const int16_t* p = net_->pst + static_cast<size_t>(feats_b_.idx[i]) * 3;
    pst_sum[0] += p[0];
    pst_sum[1] += p[1];
    pst_sum[2] += p[2];
  }
  const float inv_qpst = 1.0f / static_cast<float>(kQPst);
  for (int k = 0; k < 3; ++k) z[k] += static_cast<float>(pst_sum[k]) * inv_qpst;

  const float m = std::max(z[0], std::max(z[1], z[2]));
  const float e0 = std::exp(z[0] - m);
  const float e1 = std::exp(z[1] - m);
  const float e2 = std::exp(z[2] - m);
  const float inv = 1.0f / (e0 + e1 + e2);
  Wdl out;
  out.win = e0 * inv;
  out.draw = e1 * inv;
  out.loss = e2 * inv;
  return out;
}

}  // namespace jhbr5::nnue

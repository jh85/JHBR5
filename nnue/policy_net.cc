#include "nnue/policy_net.h"

#include "nnue/simd.h"

namespace jhbr5::nnue {

bool PolicyNet::Load(const std::string& path, std::string* err) {
  if (!file_.Load(path, err)) return false;
  const NetHeader& h = file_.header();
  if (h.kind != kNetKindPolicy) {
    *err = path + ": not a policy network";
    return false;
  }
  if (h.bucket_table_id == BucketTableId(true)) {
    see_ = true;
  } else if (h.bucket_table_id == BucketTableId(false)) {
    see_ = false;
  } else {
    *err = path + ": move bucket table id mismatch";
    return false;
  }
  version_ = static_cast<int>(h.version);
  return version_ == 2 ? LoadV2(path, err) : LoadV1(path, err);
}

bool PolicyNet::LoadV1(const std::string& path, std::string* err) {
  const NetHeader& h = file_.header();
  if (h.feature_set_id != FeatureSetId()) {
    *err = path + ": feature set id mismatch";
    return false;
  }
  l1_ = static_cast<int>(h.l1[0]);
  if (l1_ <= 0 || l1_ % 256 != 0 || h.qa != kQA || h.qb != kPolicyQB) {
    *err = path + ": unsupported policy-net dimensions";
    return false;
  }
  const auto L1 = static_cast<uint64_t>(l1_);
  const auto rows = static_cast<uint64_t>(num_rows());
  l1_w = file_.TensorAs<int8_t>("l1_w", DType::kI8, {kPolicyInputs, L1}, err);
  l1_b = file_.TensorAs<int16_t>("l1_b", DType::kI16, {L1}, err);
  out_w = file_.TensorAs<int8_t>("out_w", DType::kI8, {rows, L1 / 2}, err);
  out_b = file_.TensorAs<int16_t>("out_b", DType::kI16, {rows}, err);
  return l1_w && l1_b && out_w && out_b;
}

bool PolicyNet::LoadV2(const std::string& path, std::string* err) {
  const NetHeader& h = file_.header();
  if (h.feature_set_id != FeatureSetIdV2()) {
    *err = path + ": feature set id mismatch";
    return false;
  }
  l1_ = static_cast<int>(h.l1[0]);
  if (l1_ <= 0 || l1_ % 256 != 0 || h.qa != kQA || h.qb != kPolicyQB ||
      h.n_phase != 0) {
    *err = path + ": unsupported policy-net dimensions";
    return false;
  }
  const auto L1 = static_cast<uint64_t>(l1_);
  const auto rows = static_cast<uint64_t>(num_rows());
  l1_w = file_.TensorAs<int8_t>("l1_w", DType::kI8, {kPolicy2Inputs, L1}, err);
  l1_b = file_.TensorAs<int16_t>("l1_b", DType::kI16, {L1}, err);
  out_w = file_.TensorAs<int8_t>("out_w", DType::kI8, {rows, L1}, err);
  out_b = file_.TensorAs<int16_t>("out_b", DType::kI16, {rows}, err);
  return l1_w && l1_b && out_w && out_b;
}

PolicyScratch::PolicyScratch(const PolicyNet& net)
    : net_(&net),
      l1_(net.l1()),
      v2_(net.is_v2()),
      acc_(net.l1()),
      hl_(v2_ ? net.l1() : net.l1() / 2) {}

void PolicyScratch::ComputeHidden(const ShogiBoard& board) {
  ++expansions_;
  const size_t stride = static_cast<size_t>(l1_);
  if (v2_) {
    Policy2Features(board, &feats2_);
    for (int i = 0; i < feats2_.n; ++i) simd::Prefetch(net_->l1_w + feats2_.idx[i] * stride);
    simd::CopyBias(acc_.data(), net_->l1_b, l1_);
    simd::UpdateRows(acc_.data(), l1_, net_->l1_w, stride, feats2_.idx, feats2_.n,
                     nullptr, 0);
    rows_ += static_cast<uint64_t>(feats2_.n);
    simd::ScreluFull(acc_.data(), l1_, kQA, kPolicyShift, hl_.data());
    return;
  }
  PolicyFeatures(board, &feats_);
  for (int i = 0; i < feats_.n; ++i) simd::Prefetch(net_->l1_w + feats_.idx[i] * stride);
  simd::CopyBias(acc_.data(), net_->l1_b, l1_);
  simd::UpdateRows(acc_.data(), l1_, net_->l1_w, stride, feats_.idx, feats_.n,
                   nullptr, 0);
  rows_ += static_cast<uint64_t>(feats_.n);
  simd::PairwiseMul(acc_.data(), l1_, kQA, kPolicyShift, hl_.data());
}

int PolicyScratch::Bucket(const ShogiBoard& board, Move m) const {
  return net_->see_doubling() ? MoveBucketSee(board, m) : MoveBucket(board, m);
}

float PolicyScratch::Logit(const ShogiBoard& board, Move m) const {
  const int width = v2_ ? l1_ : l1_ / 2;
  const int bucket = Bucket(board, m);
  const int8_t* row = net_->out_w + static_cast<size_t>(bucket) * width;
  const int32_t dot = simd::DotI16I8(hl_.data(), row, width);
  const float pre = static_cast<float>(dot) / static_cast<float>(kQA * kPolicyFactor) +
                    static_cast<float>(net_->out_b[bucket]);
  return pre / static_cast<float>(kPolicyQB);
}

void PolicyScratch::Logits(const ShogiBoard& board, const MoveList& moves,
                           float* logits) {
  ComputeHidden(board);
  for (int i = 0; i < moves.size(); ++i) logits[i] = Logit(board, moves[i]);
}

}  // namespace jhbr5::nnue

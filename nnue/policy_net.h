// JHBR5 NNUE — policy network (docs/DESIGN.md §4).

#pragma once

#include <cstdint>
#include <string>

#include "nnue/aligned.h"
#include "nnue/features.h"
#include "nnue/move_buckets.h"
#include "nnue/net_format.h"
#include "nnue/types.h"

namespace jhbr5::nnue {

using lczero::MoveList;

class PolicyNet {
 public:
  bool Load(const std::string& path, std::string* err);

  int l1() const { return l1_; }
  bool see_doubling() const { return see_; }
  int num_rows() const { return see_ ? kNumBucketsSee : kNumBuckets; }

  const int8_t* l1_w = nullptr;   // [kPolicyInputs][l1]
  const int16_t* l1_b = nullptr;  // [l1]
  const int8_t* out_w = nullptr;  // [num_rows][l1 / 2]
  const int16_t* out_b = nullptr; // [num_rows]

 private:
  NetFile file_;
  int l1_ = 0;
  bool see_ = false;
};

class PolicyScratch {
 public:
  explicit PolicyScratch(const PolicyNet& net);

  // Computes the hidden vector for `board`; must precede Logit()/Logits().
  void ComputeHidden(const ShogiBoard& board);
  float Logit(const ShogiBoard& board, Move m) const;
  int Bucket(const ShogiBoard& board, Move m) const;

  // Convenience: hidden + one logit per move.
  void Logits(const ShogiBoard& board, const MoveList& moves, float* logits);

  uint64_t expansions() const { return expansions_; }
  uint64_t rows() const { return rows_; }

 private:
  const PolicyNet* net_;
  int l1_;
  AlignedBuffer<int16_t> acc_;  // [l1]
  AlignedBuffer<int16_t> hl_;   // [l1 / 2]
  FeatureList<kMaxActivePolicy> feats_;
  uint64_t expansions_ = 0, rows_ = 0;
};

}  // namespace jhbr5::nnue

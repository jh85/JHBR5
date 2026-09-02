// JHBR5 NNUE — value network (docs/DESIGN.md §3).

#pragma once

#include <cstdint>
#include <string>

#include "nnue/aligned.h"
#include "nnue/features.h"
#include "nnue/net_format.h"
#include "nnue/types.h"

namespace jhbr5::nnue {

// Shared, read-only weights.
class ValueNet {
 public:
  bool Load(const std::string& path, std::string* err);

  int l1() const { return l1_; }
  int qb() const { return qb_; }

  const int8_t* a_w = nullptr;    // [kGroupAInputs][l1]
  const int16_t* a_b = nullptr;   // [l1]
  const int8_t* b_w = nullptr;    // [kGroupBInputs][l1]
  const int16_t* b_b = nullptr;   // [l1]
  const int16_t* l2_w = nullptr;  // [kValueL2][3 * l1 / 2]
  const int16_t* l2_b = nullptr;  // [kValueL2]
  const float* l3_w = nullptr;    // [kValueL3][kValueL2]
  const float* l3_b = nullptr;    // [kValueL3]
  const float* l4_w = nullptr;    // [3][kValueL3]
  const float* l4_b = nullptr;    // [3]
  const int16_t* pst = nullptr;   // [kGroupBInputs][3]

 private:
  NetFile file_;
  int l1_ = 0;
  int qb_ = 0;
};

// Per-thread state: finny caches for the two frames plus work buffers.
class ValueScratch {
 public:
  explicit ValueScratch(const ValueNet& net);

  Wdl Evaluate(const ShogiBoard& board);

  // Forget all cached accumulators (e.g. after loading a new net).
  void Reset();

  // Row-update statistics since the last Reset() (for bench/diagnostics).
  uint64_t rows_a() const { return rows_a_; }
  uint64_t rows_b() const { return rows_b_; }
  uint64_t evals() const { return evals_; }

 private:
  const int16_t* RefreshFrame(const ShogiBoard& board, Color p);

  const ValueNet* net_;
  int l1_;
  AlignedBuffer<int16_t> cache_acc_;  // [2][81][l1]
  FrameState state_[2][81];
  AlignedBuffer<int16_t> acc_b_;      // [l1]
  AlignedBuffer<int16_t> act_;        // [3 * l1 / 2]
  FeatureList<128> adds_, subs_;
  FeatureList<kMaxActiveB> feats_b_;
  uint64_t rows_a_ = 0, rows_b_ = 0, evals_ = 0;
};

}  // namespace jhbr5::nnue

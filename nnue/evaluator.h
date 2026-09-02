// JHBR5 NNUE — the interface the search uses.
//
//   NetworkSet nets; nets.Load(value_path, policy_path, &err);   // once
//   Evaluator ev(nets);                                          // one per thread
//   Wdl wdl = ev.Evaluate(board);
//   ev.Policy(board, legal_moves, logits);

#pragma once

#include <string>

#include "nnue/policy_net.h"
#include "nnue/types.h"
#include "nnue/value_net.h"

namespace jhbr5::nnue {

struct NetworkSet {
  ValueNet value;
  PolicyNet policy;
  bool Load(const std::string& value_path, const std::string& policy_path,
            std::string* err) {
    return value.Load(value_path, err) && policy.Load(policy_path, err);
  }
};

class Evaluator {
 public:
  explicit Evaluator(const NetworkSet& nets)
      : value_(nets.value), policy_(nets.policy) {}

  Wdl Evaluate(const ShogiBoard& board) { return value_.Evaluate(board); }

  // logits[i] corresponds to moves[i]; the caller applies softmax/temperature.
  void Policy(const ShogiBoard& board, const MoveList& moves, float* logits) {
    policy_.Logits(board, moves, logits);
  }

  void Reset() { value_.Reset(); }

  ValueScratch& value() { return value_; }
  PolicyScratch& policy() { return policy_; }

 private:
  ValueScratch value_;
  PolicyScratch policy_;
};

}  // namespace jhbr5::nnue

/*
  JHBR2/JHBR3 Shogi Engine — Shared neural-network output types.

  Used by both the ONNX Runtime and TensorRT inference backends so the
  NN result struct and policy-softmax helper are defined in one place.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "shogi/board.h"
#include "shogi/encoder.h"
#include "shogi/types.h"

namespace jhbr2 {

using lczero::Color;
using lczero::Move;
using lczero::MoveList;
using lczero::ShogiBoard;
using lczero::ShogiMoveToNNIndex;

// =====================================================================
// NNOutput — result of evaluating one position
// =====================================================================

struct NNOutput {
  float value = 0.0f;    // W - L from side-to-move perspective
  float draw = 0.0f;     // Draw probability
  float wdl[3] = {0.0f, 0.0f, 0.0f};  // [Win, Draw, Loss] probabilities
  float moves_left = 0.0f;  // MLH head: model's plies-to-end estimate (0 if none)
  bool valid = true;  // false results must not be cached or backed up

  // Policy: probability for each legal move.
  // Indexed by position in the legal_moves vector passed to Evaluate().
  std::vector<float> policy;
};

// In-place softmax over `size` elements. Returns false if the resulting
// sum is not finite/positive (indicates bad input logits).
inline bool SoftmaxInPlace(float* data, int size) {
  if (size <= 0) return false;
  float max_val = data[0];
  for (int i = 1; i < size; ++i) {
    if (!std::isfinite(data[i])) return false;
    if (data[i] > max_val) max_val = data[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    data[i] = std::exp(data[i] - max_val);
    sum += data[i];
  }
  if (!std::isfinite(sum) || sum <= 0.0f) return false;
  for (int i = 0; i < size; ++i) data[i] /= sum;
  return true;
}

// Compute a softmax probability for each legal move from the raw policy
// logits. `move_to_index` is called for each legal move and must return the
// index into `logits` for that move, or -1 if the move is outside the policy
// tensor.
//
// Returns false if softmax could not be computed (e.g. non-finite logits).
// On success, `out_policy` is resized to legal_moves.size() and filled
// with probabilities that sum to 1.
template <typename MoveToIndex>
inline bool LegalPolicySoftmax(const float* logits, int policy_size,
                               const MoveList& legal_moves,
                               MoveToIndex move_to_index,
                               std::vector<float>* out_policy) {
  const size_t num_legal = legal_moves.size();
  out_policy->resize(num_legal);
  if (num_legal == 0) return true;

  float max_logit = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < num_legal; ++i) {
    const int index = move_to_index(legal_moves[i]);
    if (index >= 0 && index < policy_size) {
      (*out_policy)[i] = logits[index];
    } else {
      (*out_policy)[i] = -1000.0f;
    }
    if (!std::isfinite((*out_policy)[i])) return false;
    if ((*out_policy)[i] > max_logit) max_logit = (*out_policy)[i];
  }

  float sum = 0.0f;
  for (size_t i = 0; i < num_legal; ++i) {
    (*out_policy)[i] = std::exp((*out_policy)[i] - max_logit);
    sum += (*out_policy)[i];
  }
  if (!std::isfinite(sum) || sum <= 0.0f) return false;
  for (float& p : *out_policy) p /= sum;
  return true;
}

// Convenience overload for the standard JHBR2 model format. The model is
// always evaluated from BLACK's perspective, so WHITE's moves are flipped
// before indexing.
inline bool LegalPolicySoftmax(const float* logits, int policy_size,
                               const MoveList& legal_moves, Color side_to_move,
                               std::vector<float>* out_policy) {
  return LegalPolicySoftmax(
      logits, policy_size, legal_moves,
      [side_to_move](Move move) {
        if (side_to_move == lczero::WHITE) move.Flip();
        return ShogiMoveToNNIndex(move);
      },
      out_policy);
}

}  // namespace jhbr2

/*
  JHBR2 Shogi Engine — Neural Network Evaluator

  ONNX Runtime C++ interface for batched NN inference.
  Uses the encoder from shogi/encoder.h for input/output mapping.

  Reference: lc0 src/neural/onnx/builder.h
*/

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "inference/model_format.h"
#include "inference/nn_output.h"
#include "shogi/board.h"
#include "shogi/types.h"

namespace jhbr2 {

using lczero::MoveList;
using lczero::ShogiBoard;

// =====================================================================
// NNEvaluator — wraps ONNX Runtime for model inference
// =====================================================================

class NNEvaluator {
 public:
  // Load model from ONNX file.
  // use_gpu: create a CUDA session first, then fall back to CPU.
  // The model must have one dynamic-batch input shaped (batch, C, 9, 9).
  // num_slots is accepted for signature parity and ignored; one ORT session
  // is shared by all workers.
  explicit NNEvaluator(const std::string& onnx_path, bool use_gpu = true,
                       int device_id = 0, int num_slots = 1,
                       ModelFormat model_format = ModelFormat::kAuto);
  ~NNEvaluator();

  // Evaluate a single position.
  // legal_moves: list of legal moves in the position.
  // Returns NNOutput with value, draw, and per-move policy.
  NNOutput Evaluate(const ShogiBoard& board, const MoveList& legal_moves);

  // Evaluate a batch of positions (more efficient on GPU).
  // Each element: (board, legal_moves).
  std::vector<NNOutput> EvaluateBatch(
      const std::vector<std::pair<ShogiBoard, MoveList>>& batch);

  // Slot-aware variant provided for signature compatibility. ORT sessions
  // support concurrent Run calls, so slot_id is not needed here.
  std::vector<NNOutput> EvaluateBatchSlot(
      int slot_id,
      const std::vector<std::pair<ShogiBoard, MoveList>>& batch) {
    (void)slot_id;
    return EvaluateBatch(batch);
  }

  int num_slots() const { return 1; }

  // Whether the loaded model exposes a usable moves-left output. Search must
  // not infer support from a zero value: zero is also a valid terminal MLH
  // estimate.
  bool has_moves_left() const;

  // Is the evaluator using GPU?
  bool using_gpu() const { return using_gpu_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool using_gpu_ = false;
};

}  // namespace jhbr2

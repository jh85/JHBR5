#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "mcts/types.h"
#include "shogi/board.h"
#include "shogi/types.h"

namespace dlshogi_mcts {

struct uct_node_t;

// Owns one lazily-created child node while publishing its address atomically.
// Search workers only call get()/GetOrCreate(); tree reuse may Take()/Reset()
// slots after every search worker has joined.
class child_node_slot_t {
 public:
  child_node_slot_t() = default;
  ~child_node_slot_t();

  child_node_slot_t(const child_node_slot_t&) = delete;
  child_node_slot_t& operator=(const child_node_slot_t&) = delete;
  child_node_slot_t(child_node_slot_t&&) = delete;
  child_node_slot_t& operator=(child_node_slot_t&&) = delete;

  uct_node_t* get() const noexcept;
  explicit operator bool() const noexcept { return get() != nullptr; }

  // Concurrent callers either publish a new node or receive the node another
  // caller published. The returned pointer remains owned by this slot.
  uct_node_t* GetOrCreate();

  // Tree-lifecycle operations. These must only run after search workers join.
  std::unique_ptr<uct_node_t> Take() noexcept;
  std::unique_ptr<uct_node_t> Reset(
      std::unique_ptr<uct_node_t> replacement) noexcept;

 private:
  std::atomic<uct_node_t*> node_{nullptr};
};

// Edge statistics (dlshogi convention): `win` accumulates results from the
// perspective of the player who traversed the edge; `move_count` includes
// in-flight virtual visits.
struct child_node_t {
  child_node_t() = default;
  explicit child_node_t(lczero::Move m) : move(m) {}

  child_node_t(child_node_t&& o) noexcept
      : move(o.move),
        nnrate(o.nnrate),
        move_count(o.move_count.load(std::memory_order_relaxed)),
        win(o.win.load(std::memory_order_relaxed)),
        flags(o.flags.load(std::memory_order_relaxed)) {
    node.Reset(o.node.Take());
  }

  child_node_t& operator=(child_node_t&& o) noexcept {
    move = o.move;
    nnrate = o.nnrate;
    move_count.store(o.move_count.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    win.store(o.win.load(std::memory_order_relaxed), std::memory_order_relaxed);
    flags.store(o.flags.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
    node.Reset(o.node.Take());
    return *this;
  }

  child_node_t(const child_node_t&) = delete;
  child_node_t& operator=(const child_node_t&) = delete;

  bool IsWin() const { return flags.load(std::memory_order_acquire) & kWin; }
  bool IsLose() const { return flags.load(std::memory_order_acquire) & kLose; }
  bool IsDraw() const { return flags.load(std::memory_order_acquire) & kDraw; }
  void SetWin() { flags.fetch_or(kWin, std::memory_order_acq_rel); }
  void SetLose() { flags.fetch_or(kLose, std::memory_order_acq_rel); }
  void SetDraw() { flags.fetch_or(kDraw, std::memory_order_acq_rel); }

  // The child node (lazily created) lives in the edge itself: one array per
  // expansion, no separate slot array (JHBR5 memory reduction). Field order
  // packs the struct into 24 bytes; `flags` is public only for that reason.
  child_node_slot_t node;
  lczero::Move move;
  std::atomic<uint8_t> flags{0};
  float nnrate = 0.0f;
  std::atomic<int> move_count{0};
  std::atomic<float> win{0.0f};

 private:
  enum : uint8_t { kWin = 1, kLose = 2, kDraw = 4 };
};
static_assert(sizeof(child_node_t) <= 24, "child_node_t grew; check the layout");

// Approximate bytes held by live tree nodes (all trees in the process).
// Maintained by uct_node_t allocation/expansion/destruction.
struct TreeMemory {
  static std::atomic<size_t>& Bytes();
};

// Node state machine (docs/DESIGN.md §7.2):
//   kFresh      never visited
//   kEvaluated  value network evaluated on the first visit, no children yet
//   kExpanded   legal moves generated and policy priors assigned
struct uct_node_t {
  enum : uint8_t { kFresh = 0, kEvaluated = 1, kExpanded = 2 };

  uct_node_t();
  ~uct_node_t();

  bool IsEvaled() const {
    return state.load(std::memory_order_acquire) != kFresh;
  }
  bool IsExpanded() const {
    return state.load(std::memory_order_acquire) == kExpanded;
  }
  void SetEvaled() { state.store(kEvaluated, std::memory_order_release); }
  void SetExpanded() { state.store(kExpanded, std::memory_order_release); }

  // Allocates the child array for the legal moves of `board` (or the given
  // list). Does not change `state`; the caller publishes with SetExpanded()
  // after the priors are written.
  void ExpandNode(const lczero::ShogiBoard* board);
  void ExpandNode(const lczero::MoveList& moves);
  void InitChildNodes() {}  // kept for API compatibility; slots live in `child`
  uct_node_t* CreateChildNode(int i) { return child[i].node.GetOrCreate(); }
  void CreateSingleChildNode(lczero::Move move);
  uct_node_t* ReleaseChildrenExceptOne(lczero::Move move);

  std::atomic<uint8_t> state{kFresh};
  std::atomic<int> move_count{0};
  std::atomic<float> win{0.0f};
  std::atomic<float> visited_nnrate{0.0f};
  short child_num = 0;
  std::unique_ptr<child_node_t[]> child;
};

class NodeTree {
 public:
  NodeTree();
  ~NodeTree();

  // Returns true when the new move history extends the previous root.
  bool ResetToPosition(uint64_t starting_pos_key,
                       const std::vector<lczero::Move>& moves);
  uct_node_t* GetCurrentHead() const { return current_head_; }
  void DeallocateTree();

 private:
  uct_node_t* current_head_ = nullptr;
  std::unique_ptr<uct_node_t> gamebegin_node_;
  uint64_t history_starting_pos_key_ = 0;
  std::vector<lczero::Move> current_position_moves_;
  bool has_position_ = false;
};

}  // namespace dlshogi_mcts

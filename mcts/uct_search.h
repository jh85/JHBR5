// JHBR5 — Monty-style MCTS on JHBR3's dlshogi tree with CPU NNUE evaluation.
//
// One iteration = descend by PUCT, evaluate the leaf synchronously with the
// value network on its first visit, generate moves + policy priors on its
// second visit (docs/DESIGN.md §7), back up immediately. No leaf batching.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "mcts/butterfly.h"
#include "mcts/search_primitives.h"
#include "mcts/uct_node.h"
#include "nnue/eval_cache.h"
#include "nnue/evaluator.h"
#include "usi/time_manager.h"

namespace dlshogi_mcts {

struct SearchInfo {
  int depth = 0;
  int score_cp = 0;
  int nodes = 0;
  int nps = 0;
  int time_ms = 0;
  std::vector<lczero::Move> pv;
  jhbr5::nnue::EvalCache::Stats cache;
};
using InfoCallback = std::function<void(const SearchInfo&)>;
using SearchStartedCallback = std::function<void()>;

struct SearchConfig {
  float c_init = 1.25f;
  float c_base = 19652.0f;
  float c_fpu_reduction = 0.27f;
  float c_init_root = 1.25f;
  float c_base_root = 19652.0f;
  float c_fpu_reduction_root = 0.0f;
  float draw_value_black = 0.5f;
  float draw_value_white = 0.5f;
  float resign_threshold = 0.01f;
  int max_nodes = 100000000;
  float max_time = 0.0f;
  jhbr2::TimeBudget time_budget;
  int threads = 1;
  int max_moves_to_draw = 100000;
  int leaf_mate_depth = 5;

  // Before returning a move, reject root candidates that let the opponent
  // force mate within this many plies.
  int root_mate_depth = 7;

  size_t eval_cache_mb = 64;
  size_t tree_memory_mb = 4096;

  // Monty extras, off by default until SPRT-tested (docs/DESIGN.md §7.8).
  bool use_butterfly = false;
  int butterfly_divisor = 17179;
  int butterfly_reduction = 8358;
  bool use_policy_temperature = false;
  float pst_root = 0.3349f;
  float pst_depth = 1.5777f;
  float pst_win_threshold = 0.5655f;
  float pst_win_max = 1.6260f;
  float pst_base = 0.0960f;
  float draw_scale = 0.0f;
  float draw_quadratic = 0.0f;

  int info_interval_ms = 1000;
  InfoCallback info_callback = nullptr;
};

struct SearchResult {
  lczero::Move best_move;
  bool tree_reused = false;
  int root_visits_before = 0;
  int nodes = 0;
  float time_sec = 0.0f;
  float nps = 0.0f;
  int score_cp = 0;
  std::vector<lczero::Move> pv;
  uint64_t evals = 0;
  uint64_t expansions = 0;
  jhbr5::nnue::EvalCache::Stats cache;
  jhbr2::TimeBudget time_budget;
  jhbr2::AdaptiveTimeDecision time_decision;
  bool root_guard_cancelled = false;
};

class UCTSearcher;

class Search {
 public:
  using Clock = std::chrono::steady_clock;

  Search(const jhbr5::nnue::NetworkSet* nets, const SearchConfig& config);
  ~Search();

  SearchResult Run(lczero::ShogiBoard board, uint64_t starting_pos_key,
                   const std::vector<lczero::Move>& moves,
                   Clock::time_point move_start = Clock::now(),
                   SearchStartedCallback on_search_started = nullptr);
  // Called from the acknowledged isready phase. Clears game-specific tree
  // state, the evaluation cache and the history table.
  void PrepareForNewGame();
  void Stop() { stop_.store(true, std::memory_order_release); }
  void SetMaxTime(float seconds) { config_.max_time = seconds; }
  void SetMaxNodes(size_t n) { config_.max_nodes = static_cast<int>(n); }
  void SetTimeBudget(const jhbr2::TimeBudget& budget) {
    config_.time_budget = budget;
    config_.max_time = budget.mcts_time_seconds;
  }
  const SearchConfig& config() const { return config_; }

 private:
  friend class UCTSearcher;

  bool IsSearchActive() const;
  void ExpandRoot();
  void RejectRootMates();
  jhbr2::RootSearchSnapshot CaptureRootSnapshot() const;
  void MaybeManageTime(bool force = false);
  unsigned SelectBestChild(const uct_node_t* node) const;
  SearchResult BuildResult() const;
  std::vector<lczero::Move> GetPV() const;
  int QToCentipawns(float win_rate) const;
  void MaybeOutputInfo();

  SearchConfig config_;
  const jhbr5::nnue::NetworkSet* nets_;
  std::vector<std::unique_ptr<UCTSearcher>> searchers_;
  NodeTree tree_;
  jhbr5::nnue::EvalCache eval_cache_;
  ButterflyTable butterfly_;
  lczero::ShogiBoard root_board_;
  uct_node_t* root_ = nullptr;
  bool tree_reused_ = false;
  int root_visits_before_ = 0;
  std::atomic<bool> stop_{false};
  std::atomic<bool> adaptive_stop_{false};
  std::atomic<bool> tree_full_{false};
  std::atomic<int> playout_count_{0};
  std::atomic<uint64_t> evals_{0};
  std::atomic<uint64_t> expansions_{0};
  Timer timer_;
  jhbr2::AdaptiveTimeController time_controller_;
  std::atomic<int> last_time_check_ms_{-1000000};
  std::atomic<bool> time_check_busy_{false};
  bool root_guard_cancelled_ = false;
  mutable std::mutex info_mutex_;
  int last_info_ms_ = 0;
};

}  // namespace dlshogi_mcts

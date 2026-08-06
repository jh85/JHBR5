#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "mcts/search_primitives.h"
#include "mcts/uct_node.h"
#include "inference/nn_cache.h"
#include "usi/time_manager.h"
#ifdef USE_TENSORRT
#include "inference/nn_tensorrt.h"
#else
#include "inference/nn_eval.h"
#endif

namespace dlshogi_mcts {

struct SearchInfo {
  int depth = 0;
  int score_cp = 0;
  int nodes = 0;
  int nps = 0;
  int time_ms = 0;
  std::vector<lczero::Move> pv;
  jhbr2::NNCacheStats nn_cache;
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
  int max_nodes = 800;
  float max_time = 0.0f;
  jhbr2::TimeBudget time_budget;
  int workers_per_gpu = 2;
  int minibatch_size = 128;
  int max_moves_to_draw = 100000;
  int leaf_mate_depth = 5;

  // Before returning a move, reject root candidates that let the opponent
  // force mate within this many plies. This makes deeper defensive coverage
  // affordable without paying for it at every MCTS leaf.
  int root_mate_depth = 7;

  // Lc0-style moves-left (MLH) effect in selection. Keep opt-in until the
  // lc0-shaped defaults have been strength-tested for Shogi; enabling is also
  // gated by every active evaluator actually exposing an MLH output.
  MovesLeftParameters moves_left;

  size_t nn_cache_size = 0;
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
  jhbr2::NNCacheStats nn_cache;
  jhbr2::TimeBudget time_budget;
  jhbr2::AdaptiveTimeDecision time_decision;
  bool root_guard_cancelled = false;
};

class Search;

class UCTSearcherGroup {
 public:
  UCTSearcherGroup(Search* owner, jhbr2::NNEvaluator* nn, int gpu_id,
                   int threads, int batch_max);
  UCTSearcherGroup(UCTSearcherGroup&&) noexcept;
  UCTSearcherGroup& operator=(UCTSearcherGroup&&) noexcept;
  ~UCTSearcherGroup();

  void Run();
  void Join();

  Search* owner = nullptr;
  jhbr2::NNEvaluator* nn = nullptr;
  int gpu_id = 0;

 private:
  std::vector<std::unique_ptr<class UCTSearcher>> searchers_;
};

class Search {
 public:
  using Clock = std::chrono::steady_clock;

  Search(std::vector<jhbr2::NNEvaluator*> evaluators, const SearchConfig& config);
  ~Search();

  SearchResult Run(lczero::ShogiBoard board, uint64_t starting_pos_key,
                   const std::vector<lczero::Move>& moves,
                   Clock::time_point move_start = Clock::now(),
                   SearchStartedCallback on_search_started = nullptr);
  // Called from the acknowledged isready phase. Clears game-specific tree
  // state and NN entries while preserving GPU evaluators, workers, and cache
  // bucket allocation.
  void PrepareForNewGame();
  void Stop() { stop_.store(true, std::memory_order_release); }
  void SetMaxTime(float seconds) { config_.max_time = seconds; }
  void SetMaxNodes(size_t n) { config_.max_nodes = static_cast<int>(n); }
  void SetTimeBudget(const jhbr2::TimeBudget& budget) {
    config_.time_budget = budget;
    config_.max_time = budget.mcts_time_seconds;
  }

 private:
  friend class UCTSearcher;
  friend class UCTSearcherGroup;

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
  std::vector<jhbr2::NNEvaluator*> evaluators_;
  std::vector<UCTSearcherGroup> groups_;
  NodeTree tree_;
  jhbr2::NNCache nn_cache_;
  lczero::ShogiBoard root_board_;
  uct_node_t* root_ = nullptr;
  bool tree_reused_ = false;
  int root_visits_before_ = 0;
  std::atomic<bool> stop_{false};
  std::atomic<bool> adaptive_stop_{false};
  std::atomic<int> playout_count_{0};
  Timer timer_;
  jhbr2::AdaptiveTimeController time_controller_;
  std::atomic<int> last_time_check_ms_{-1000000};
  std::atomic<bool> time_check_busy_{false};
  int in_flight_playouts_ = 0;
  bool root_guard_cancelled_ = false;
  bool moves_left_supported_ = false;
  mutable std::mutex info_mutex_;
  int last_info_ms_ = 0;
};

}  // namespace dlshogi_mcts
